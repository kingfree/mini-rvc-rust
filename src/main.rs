use axum::{
    routing::{get, post},
    extract::{Multipart, State, WebSocketUpgrade, ws::{WebSocket, Message}},
    Json, Router,
};
use std::net::SocketAddr;
use tower_http::services::ServeDir;
use tower_http::cors::CorsLayer;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::Instant;
use candle_core::{Device, Tensor};
use tract_onnx::prelude::{* , tvec, Framework};
use tokio::sync::Mutex;

mod audio_processing;
mod feature_extractor;
mod pitch_extractor;
mod audio_stitching;
mod realtime_pipeline;
mod ring_buffer;

use realtime_pipeline::RvcPipeline;

#[derive(Serialize)]
struct ExtractionResponse {
    success: bool,
    message: String,
    time_ms: u64,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct WsConfig {
    pitch_shift: f32,
    #[serde(default)]
    model_id: Option<String>,
}

struct AppState {
    pipeline: Mutex<Option<RvcPipeline>>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    name: String,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--test" {
        return run_component_test();
    }
    
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_server())
}

async fn run_server() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let device = if candle_core::utils::metal_is_available() {
        println!("🚀 检测到 Metal 加速支持，正在启用 GPU 推理...");
        Device::new_metal(0)?
    } else {
        println!("⚠️ 未检测到 Metal，将使用 CPU 进行推理 (可能会很慢)");
        Device::Cpu
    };
    
    let content_vec_path = "pretrain/content_vec_500.onnx";
    let rmvpe_path = "pretrain/rmvpe.onnx";
    
    // Default model
    let rvc_path = "models/Anon_v2_merged.onnx";

    let pipeline = if std::path::Path::new(content_vec_path).exists() && 
                      std::path::Path::new(rmvpe_path).exists() {
        
        let initial_model = if std::path::Path::new(rvc_path).exists() {
             rvc_path.to_string()
        } else {
            // Find first available model
            if let Ok(mut entries) = std::fs::read_dir("models") {
                entries.find_map(|entry| {
                    let path = entry.ok()?.path();
                    if path.extension()?.to_str()? == "onnx" {
                        Some(path.to_str()?.to_string())
                    } else {
                        None
                    }
                }).unwrap_or_else(|| "".to_string())
            } else {
                "".to_string()
            }
        };

        if !initial_model.is_empty() {
             println!("正在初始化实时推理管道，使用模型: {}...", initial_model);
             match RvcPipeline::new(content_vec_path, rmvpe_path, &initial_model, device) {
                Ok(p) => Some(p),
                Err(e) => {
                    println!("警告: 管道初始化失败: {}", e);
                    None
                }
            }
        } else {
             println!("警告: 未找到任何 RVC 模型文件 (models/*.onnx)。");
             None
        }
    } else {
        println!("警告: 未找到必要的预训练模型文件 (ContentVec/RMVPE)，实时功能将受限。");
        None
    };

    let state = Arc::new(AppState {
        pipeline: Mutex::new(pipeline),
    });

    let app = Router::new()
        .route("/api/health", get(|| async { "OK" }))
        .route("/api/models", get(list_models))
        .route("/ws", get(ws_handler))
        .with_state(state)
        .fallback_service(ServeDir::new("web/dist"))
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([127, 0, 0, 1], 3001));
    println!("--- Mini RVC Rust Server Running on http://{} ---", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn list_models() -> Json<Vec<ModelInfo>> {
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "onnx" {
                    if let Some(stem) = path.file_stem() {
                        let id = path.file_name().unwrap().to_string_lossy().to_string();
                        let name = stem.to_string_lossy().to_string();
                        models.push(ModelInfo { id, name });
                    }
                }
            }
        }
    }
    // Sort for consistency
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Json(models)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    println!("WebSocket 客户端已连接");
    
    let mut buffer = ring_buffer::RingBuffer::new(16000 * 5); // 5 seconds buffer
    let window_size = 16000; // 1 second window for processing
    let hop_size = 8000;    // 0.5 second hop
    let mut pitch_shift = 0.0;

    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Binary(data) => {
                // Convert bytes (f32 le) to f32 slice
                let samples: Vec<f32> = data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                
                buffer.push(&samples);

                // Process if we have enough data
                while buffer.len() >= window_size {
                    if let Some(chunk) = buffer.read_and_advance(window_size, hop_size) {
                        let mut pipeline_guard = state.pipeline.lock().await;
                        if let Some(ref mut pipeline) = *pipeline_guard {
                                                            match pipeline.process(&chunk, pitch_shift) {
                                                            Ok(output) => {
                                                                if output.is_empty() {
                                                                    println!("Warning: Output is empty!");
                                                                } else {
                                                                    let max_val = output.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                                                                    if max_val == 0.0 {
                                                                        println!("Warning: Output is SILENCE (all zeros)!");
                                                                    } else {
                                                                        println!("Output chunk: len={}, max_amp={:.4}", output.len(), max_val);
                                                                    }
                                                                }
                            
                                                                // Convert output f32 to bytes
                                                                let mut out_bytes = Vec::with_capacity(output.len() * 4);
                                                                for s in output {
                                                                    out_bytes.extend_from_slice(&s.to_le_bytes());
                                                                }
                                                                if socket.send(Message::Binary(out_bytes)).await.is_err() {
                                                                    break;
                                                                }
                                                            }                                Err(e) => {
                                    eprintln!("推理错误: {}", e);
                                }
                            }
                        }
                    }
                }
            }
            Message::Text(text) => {
                if let Ok(config) = serde_json::from_str::<WsConfig>(&text) {
                    pitch_shift = config.pitch_shift;
                    // Handle Model Switching
                    if let Some(model_id) = config.model_id {
                        let model_path = format!("models/{}", model_id);
                        if std::path::Path::new(&model_path).exists() {
                            let mut pipeline_guard = state.pipeline.lock().await;
                            if let Some(ref mut pipeline) = *pipeline_guard {
                                println!("正在切换模型: {} ...", model_id);
                                if let Err(e) = pipeline.set_model(&model_path) {
                                    eprintln!("切换模型失败: {}", e);
                                } else {
                                    println!("模型切换成功！");
                                }
                            }
                        } else {
                            eprintln!("请求的模型不存在: {}", model_path);
                        }
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    println!("WebSocket 客户端断开连接");
}

async fn handle_extract(
    extractor: Option<Arc<feature_extractor::ContentVec>>,
    mut multipart: Multipart,
) -> Json<ExtractionResponse> {
    let Some(extractor) = extractor else {
        return Json(ExtractionResponse {
            success: false,
            message: "Model not loaded".to_string(),
            time_ms: 0,
            shape: vec![],
        });
    };

    while let Ok(Some(field)) = multipart.next_field().await {
        if let Some(name) = field.name() {
            if name == "file" {
                let data = field.bytes().await.unwrap();
                
                let start = Instant::now();
                match feature_extractor::load_wav_memory(&data) {
                    Ok((waveform, sr)) => {
                        let waveform_16k = audio_processing::resample_to_16k(&waveform, sr);
                        
                        match extractor.extract(&waveform_16k) {
                            Ok(features) => {
                                let duration = start.elapsed().as_millis() as u64;
                                return Json(ExtractionResponse {
                                    success: true,
                                    message: format!("特征提取成功 (采样率: {}Hz)", sr),
                                    time_ms: duration,
                                    shape: features.shape().to_vec(),
                                });
                            }
                            Err(e) => {
                                return Json(ExtractionResponse {
                                    success: false,
                                    message: format!("推理失败: {}", e),
                                    time_ms: 0,
                                    shape: vec![],
                                });
                            }
                        }
                    }
                    Err(e) => {
                        return Json(ExtractionResponse {
                            success: false,
                            message: format!("WAV 解析失败: {}", e),
                            time_ms: 0,
                            shape: vec![],
                        });
                    }
                }
            }
        }
    }

    Json(ExtractionResponse {
        success: false,
        message: "No file uploaded".to_string(),
        time_ms: 0,
        shape: vec![],
    })
}

fn post_process_rmvpe(f0_probs: &Tensor) -> anyhow::Result<Tensor> {
    // f0_probs: [1, n_frames, 360]
    let (n_batch, n_frames, n_bins) = f0_probs.dims3()?;
    
    // 1. Argmax over bins
    let argmax = f0_probs.argmax(2)?; // [1, n_frames]
    let argmax_data = argmax.to_vec2::<u32>()?;
    
    // 2. Map bin index to frequency
    // f = 10 * 2 ^ ((bin * 20 + 1991.303504) / 1200)
    let mut f0_values = Vec::with_capacity(n_batch * n_frames);
    for batch in argmax_data {
        for bin in batch {
            let cents = (bin as f32) * 20.0 + 1991.303504;
            let f0 = 10.0 * 2.0f32.powf(cents / 1200.0);
            // Thresholding: if the probability is too low, set to 0 (unvoiced)
            // For now, let's keep it simple.
            f0_values.push(f0);
        }
    }
    
    Tensor::from_vec(f0_values, (n_batch, n_frames), f0_probs.device()).map_err(anyhow::Error::from)
}

fn pad_tensor(t: &Tensor, target_len: usize) -> anyhow::Result<Tensor> {
    let (b, cur_len, d) = t.dims3()?;
    if cur_len >= target_len {
        return Ok(t.narrow(1, 0, target_len)?);
    }
    
    let pad_len = target_len - cur_len;
    let padding = Tensor::zeros((b, pad_len, d), t.dtype(), t.device())?;
    Tensor::cat(&[t, &padding], 1).map_err(anyhow::Error::from)
}

fn downsample_f0(f0: &Tensor, target_len: usize) -> anyhow::Result<Tensor> {
    let (b, cur_len) = f0.dims2()?;
    if cur_len == target_len {
        return Ok(f0.clone());
    }
    
    // Simple decimation or averaging. Here we use decimation for simplicity.
    let stride = cur_len as f32 / target_len as f32;
    let mut indices = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let idx = (i as f32 * stride).floor() as u32;
        indices.push(idx);
    }
    
    let indices_tensor = Tensor::new(indices, f0.device())?;
    f0.index_select(&indices_tensor, 1).map_err(anyhow::Error::from)
}

fn run_component_test() -> anyhow::Result<()> {
    // ... (previous code)
    println!("--- RVC Rust Inference Components Test ---");
    let device = if candle_core::utils::metal_is_available() {
        println!("🚀 检测到 Metal 加速支持，正在启用 GPU 推理 (Test)...");
        Device::new_metal(0)?
    } else {
        println!("⚠️ 未检测到 Metal，将使用 CPU 进行推理 (Test)...");
        Device::Cpu
    };

    let wav_path = "assets/test.wav";
    if !std::path::Path::new(wav_path).exists() {
        println!("错误: 未找到音频文件 {}。", wav_path);
        return Ok(());
    }
    
    let (waveform, sr) = feature_extractor::load_wav(wav_path)?;
    println!("音频载入成功，采样率: {}, 总采样点: {}", sr, waveform.len());

    let waveform_16k_full = audio_processing::resample_to_16k(&waveform, sr);
    // Pad or truncate to exactly 20320 samples to get 128 frames for RMVPE
    // n_frames = (padded_len - n_fft) / hop + 1 = (20320 + 1024 - 1024) / 160 + 1 = 128
    let target_samples = 20320;
    let mut waveform_16k = waveform_16k_full.clone();
    if waveform_16k.len() > target_samples {
        waveform_16k.truncate(target_samples);
    } else if waveform_16k.len() < target_samples {
        waveform_16k.resize(target_samples, 0.0);
    }
    println!("已重采样并调整至 16kHz ({} samples), 采样点数: {}", target_samples, waveform_16k.len());

    let hubert_path = "pretrain/content_vec_500.onnx";
    let mut features_candle: Option<Tensor> = None;
    if std::path::Path::new(hubert_path).exists() {
        println!("正在加载 ContentVec (Tract): {}...", hubert_path);
        let extractor = feature_extractor::ContentVec::new(hubert_path)?;
        let start = Instant::now();
        let features = extractor.extract(&waveform_16k)?;
        println!("✅ ContentVec 提取成功！耗时: {:?}, 形状: {:?}", start.elapsed(), features.shape());

        // Convert tract Tensor to candle Tensor
        let shape: Vec<usize> = features.shape().to_vec();
        let data: &[f32] = features.as_slice::<f32>()?;
        features_candle = Some(Tensor::from_slice(data, shape, &device)?);
        drop(features);
        drop(extractor);
        println!("   ContentVec 模型已释放");
    }

    let rmvpe_path = "pretrain/rmvpe.onnx";
    let mut f0_tensor: Option<Tensor> = None;
    if std::path::Path::new(rmvpe_path).exists() {
        println!("正在加载 RMVPE 模型 (Tract)...");

        let mel_extractor = audio_processing::MelSpectrogram::new(16000, 1024, 160, 1024, 128);
        println!("计算 Mel 频谱...");
        let mel = mel_extractor.forward(&waveform_16k)?;
        println!("   Mel 频谱形状: {:?}", mel.dim());

        let (n_mels, n_frames) = mel.dim();
        let mel_flat: Vec<f32> = mel.iter().cloned().collect();

        {
            // Use Tract for RMVPE inference, scoped to free model after use
            let rmvpe_model = tract_onnx::onnx()
                .model_for_path(rmvpe_path)?
                .with_input_fact(0, f32::fact([1, n_mels, n_frames]).into())?
                .into_optimized()?
                .into_runnable()?;

            let mel_tract = tract_onnx::prelude::Tensor::from_shape(&[1, n_mels, n_frames], &mel_flat)?;

            println!("执行 RMVPE 推理 (Tract)...");
            let start_rmvpe = Instant::now();
            match rmvpe_model.run(tvec!(mel_tract.into())) {
                Ok(outputs) => {
                    println!("✅ RMVPE 推理成功 (Tract)！耗时: {:?}", start_rmvpe.elapsed());

                    // Get the first output (f0 probabilities)
                    let f0_probs = outputs[0].to_array_view::<f32>()?;
                    println!("   F0 概率形状: {:?}", f0_probs.shape());

                    // Convert to candle tensor
                    let f0_shape: Vec<usize> = f0_probs.shape().to_vec();
                    let f0_data: Vec<f32> = f0_probs.iter().cloned().collect();
                    f0_tensor = Some(Tensor::from_vec(f0_data, f0_shape, &device)?);
                }
                Err(e) => {
                    println!("❌ RMVPE 推理失败 (Tract): {:?}", e);
                }
            }
        }
        println!("   RMVPE 模型已释放");
    }

    // 4. 加载角色模型
    let anon_path = "models/Yukina_v2_merged.onnx";
    if std::path::Path::new(anon_path).exists() && features_candle.is_some() && f0_tensor.is_some() {
        println!("正在加载角色模型 (Yukina v2)...");
        
        // 我们期望的形状是 [1, 64, 768]
        let feats = features_candle.unwrap();
        println!("原始特征形状: {:?}", feats.shape());
        let feats = pad_tensor(&feats, 64)?;
        println!("处理后特征形状: {:?}", feats.shape());

        let f0_probs = f0_tensor.unwrap();
        println!("原始 RMVPE 概率形状: {:?}", f0_probs.shape());
        let f0_processed = post_process_rmvpe(&f0_probs)?;
        println!("处理后 F0 形状 (100Hz): {:?}", f0_processed.shape());
        let f0_downsampled = downsample_f0(&f0_processed, 64)?;
        println!("重采样后 F0 形状 (50Hz): {:?}", f0_downsampled.shape());
        
        // We expect feats [1, 64, 768]
        let t = feats.dim(1)?;
        // Keep p_len as rank 1 (confirmed by error message)
        let p_len = Tensor::from_slice(&[t as i64], (1,), &device)?;
        // Revert sid to rank 1
        let sid = Tensor::from_slice(&[0i64], (1,), &device)?;
        
        // Keep pitchf as rank 2 [1, 64] to match model expectations
        let pitchf = f0_downsampled.clone();

        // RVC pitch formula: pitch = 12 * log2(f0 / 10)
        let f0_data = f0_downsampled.to_vec2::<f32>()?;
        let pitch_indices: Vec<i64> = f0_data[0].iter().map(|&f| {
            if f > 0.0 {
                let pitch = 12.0 * (f / 10.0).log2();
                (pitch.round() as i64).clamp(1, 255)
            } else {
                0
            }
        }).collect();
        // Rank 2 [1, 64] to match model expectations
        let pitch = Tensor::from_slice(&pitch_indices, (1, pitch_indices.len()), &device)?;
        println!("Pitch 范围: {:?} - {:?}", pitch_indices.iter().min(), pitch_indices.iter().max());

        let mut inputs = std::collections::HashMap::new();
        inputs.insert("feats".to_string(), feats.clone());
        inputs.insert("p_len".to_string(), p_len);
        inputs.insert("pitch".to_string(), pitch.clone());
        inputs.insert("pitchf".to_string(), pitchf.clone());
        inputs.insert("sid".to_string(), sid);

        println!("Inputs shapes:");
        for (k, v) in &inputs {
            println!("  {}: shape={:?}, rank={}", k, v.shape(), v.shape().dims().len());
        }

        println!("执行角色模型推理 (Candle)...");
        let start_vc = Instant::now();

        let model_proto = candle_onnx::read_file(anon_path)?;
        match candle_onnx::simple_eval(&model_proto, inputs) {
            Ok(outputs) => {
                println!("✅ 角色模型推理成功 (Candle)！耗时: {:?}", start_vc.elapsed());
                if let Some(audio) = outputs.get("audio") {
                    println!("   输出音频形状: {:?}", audio.shape());
                } else {
                    println!("   输出 keys: {:?}", outputs.keys().collect::<Vec<_>>());
                }
            }
            Err(e) => {
                println!("❌ 角色模型推理失败 (Candle): {:?}", e);
            }
        }
    }

    println!("--- 组件测试完成 ---");
    Ok(())
}
