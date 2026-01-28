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
use tokio::sync::mpsc::{self, Sender};
use futures::{sink::SinkExt, stream::StreamExt};

mod audio_processing;
mod feature_extractor;
mod pitch_extractor;
mod audio_stitching;
mod realtime_pipeline;
mod ring_buffer;
mod rvc_engine;

use realtime_pipeline::{RvcPipeline, InferenceBackend};

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
    #[serde(default)]
    index_rate: f32,
}

struct InferenceRequest {
    audio_chunk: Vec<f32>,
    pitch_shift: f32,
    index_rate: f32,
    hop_size: usize,
    timestamp: Instant,
    sender: Sender<InferenceResult>,
}

struct InferenceResult {
    audio_data: Vec<f32>,
    latency: u128,
}

struct AppState {
    tx: Sender<InferenceRequest>,
    model_path_tx: Sender<String>,
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
    let rvc_path = "models/Yukina_v2_merged.onnx"; 
    // We assume index file is named similarly or specified. 
    // For simplicity, let's look for "models/Yukina_v2_index.safetensors"
    let index_path = "models/Yukina_v2_index.safetensors";

    // Channels for inference worker
    let (tx, mut rx) = mpsc::channel::<InferenceRequest>(100);
    let (model_path_tx, mut model_path_rx) = mpsc::channel::<String>(10);

    // Spawn Inference Worker
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            println!("🔊 推理线程已启动");
            
            let mut pipeline = if std::path::Path::new(content_vec_path).exists() && 
                                  std::path::Path::new(rmvpe_path).exists() {
                let initial_model = if std::path::Path::new(rvc_path).exists() {
                     rvc_path.to_string()
                } else {
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

                // Check for corresponding index file
                let initial_index = if !initial_model.is_empty() {
                    let p = std::path::Path::new(&initial_model);
                    let stem = p.file_stem().unwrap().to_str().unwrap();
                    // Clean "_merged" suffix if present
                    let base_name = stem.replace("_merged", "");
                    let idx_path = format!("models/{}_index.safetensors", base_name);
                    if std::path::Path::new(&idx_path).exists() {
                        Some(idx_path)
                    } else {
                        None
                    }
                } else {
                    None
                };

                if !initial_model.is_empty() {
                     println!("正在初始化实时推理管道，使用模型: {}...", initial_model);
                     if let Some(ref idx) = initial_index {
                         println!("  加载索引: {}", idx);
                     }

                     // Select backend based on features
                     let backend = if cfg!(feature = "onnxruntime") {
                         #[cfg(feature = "onnxruntime")]
                         {
                             InferenceBackend::OnnxRuntime
                         }
                         #[cfg(not(feature = "onnxruntime"))]
                         {
                             InferenceBackend::Candle
                         }
                     } else {
                         InferenceBackend::Candle
                     };
                     
                     println!("  使用推理后端: {:?} ({})", backend, if matches!(device, Device::Cpu) { "CPU" } else { "Accelerated" });

                     match RvcPipeline::new(content_vec_path, rmvpe_path, &initial_model, initial_index.as_ref(), device.clone(), backend) {
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
                println!("警告: 未找到必要的预训练模型文件，实时功能将受限。");
                None
            };

            loop {
                tokio::select! {
                    Some(req) = rx.recv() => {
                        if let Some(p) = &mut pipeline {
                            // Process inference
                            match p.process(&req.audio_chunk, req.hop_size, req.pitch_shift, req.index_rate) {
                                Ok(output) => {
                                    let total_latency = req.timestamp.elapsed().as_millis();
                                    
                                    if total_latency > 500 {
                                        println!("⚠️ 高延迟警告: 总延迟 {}ms", total_latency);
                                    }

                                    let result = InferenceResult {
                                        audio_data: output,
                                        latency: total_latency,
                                    };
                                    let _ = req.sender.send(result).await;
                                }
                                Err(e) => {
                                    eprintln!("推理错误: {}", e);
                                }
                            }
                        }
                    }
                    Some(new_model_path) = model_path_rx.recv() => {
                        if let Some(p) = &mut pipeline {
                            println!("Worker: Switching model to {}...", new_model_path);
                            // Select backend based on features
                            let backend = if cfg!(feature = "onnxruntime") {
                                #[cfg(feature = "onnxruntime")]
                                {
                                    InferenceBackend::OnnxRuntime
                                }
                                #[cfg(not(feature = "onnxruntime"))]
                                {
                                    InferenceBackend::Candle
                                }
                            } else {
                                InferenceBackend::Candle
                            };

                            if let Err(e) = p.set_model(&new_model_path, backend) {
                                eprintln!("Worker: Failed to switch model: {}", e);
                            } else {
                                // Try to find and load new index
                                let path = std::path::Path::new(&new_model_path);
                                if let Some(stem) = path.file_stem() {
                                    let base_name = stem.to_str().unwrap().replace("_merged", "");
                                    let idx_path = format!("models/{}_index.safetensors", base_name);
                                    let _ = p.set_index(&idx_path); // Ignore error if index missing
                                }
                                println!("Worker: Model switched successfully.");
                            }
                        }
                    }
                }
            }
        });
    });

    let hubert_path = "pretrain/content_vec_500.onnx";
    let extractor = if std::path::Path::new(hubert_path).exists() {
        Some(Arc::new(feature_extractor::ContentVec::new(hubert_path)?))
    } else {
        None
    };

    let state = Arc::new(AppState {
        tx,
        model_path_tx,
    });

    let app = Router::new()
        .route("/api/health", get(|| async { "OK" }))
        .route("/api/models", get(list_models))
        .route("/api/extract", post(move |multipart| handle_extract(extractor, multipart)))
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
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Json(models)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    println!("WebSocket 客户端已连接");
    let (mut sender, mut receiver) = socket.split();
    
    let mut buffer = ring_buffer::RingBuffer::new(16000 * 5); 
    let window_size = 6400; 
    let hop_size = 4800;    
    let mut pitch_shift = 0.0;
    let mut index_rate = 0.3; // Default index rate

    // Response channel
    let (resp_tx, mut resp_rx) = mpsc::channel::<InferenceResult>(20);

    // Spawn a task to send responses back to the socket
    let mut send_task = tokio::spawn(async move {
        while let Some(result) = resp_rx.recv().await {
            let output = result.audio_data;
            if !output.is_empty() {
                let len = output.len();
                let mut out_bytes = Vec::with_capacity(len * 4);
                for s in output {
                    out_bytes.extend_from_slice(&s.to_le_bytes());
                }
                if sender.send(Message::Binary(out_bytes)).await.is_err() {
                    break;
                }
                
                // Send metadata (latency)
                let meta = serde_json::json!({
                    "latency": result.latency,
                    "len": len
                });
                if sender.send(Message::Text(meta.to_string())).await.is_err() {
                    break;
                }
            }
        }
    });

    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Binary(data) => {
                let samples: Vec<f32> = data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                
                buffer.push(&samples);

                while buffer.len() >= window_size {
                    if let Some(chunk) = buffer.read_and_advance(window_size, hop_size) {
                        let req = InferenceRequest {
                            audio_chunk: chunk,
                            pitch_shift,
                            index_rate,
                            hop_size,
                            timestamp: Instant::now(),
                            sender: resp_tx.clone(),
                        };
                        
                        if state.tx.send(req).await.is_err() {
                            break;
                        }
                    }
                }
            }
            Message::Text(text) => {
                if let Ok(config) = serde_json::from_str::<WsConfig>(&text) {
                    pitch_shift = config.pitch_shift;
                    index_rate = config.index_rate; // Update index rate
                    
                    if let Some(model_id) = config.model_id {
                        let model_path = format!("models/{}", model_id);
                        if std::path::Path::new(&model_path).exists() {
                            let _ = state.model_path_tx.send(model_path).await;
                        }
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    send_task.abort();
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
                                    message: format!("Success (SR: {}Hz)", sr),
                                    time_ms: duration,
                                    shape: features.shape().to_vec(),
                                });
                            }
                            Err(e) => return Json(ExtractionResponse { success: false, message: e.to_string(), time_ms: 0, shape: vec![] }),
                        }
                    }
                    Err(e) => return Json(ExtractionResponse { success: false, message: e.to_string(), time_ms: 0, shape: vec![] }),
                }
            }
        }
    }
    Json(ExtractionResponse { success: false, message: "No file".to_string(), time_ms: 0, shape: vec![] })
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
    let stride = cur_len as f32 / target_len as f32;
    let mut indices = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let idx = (i as f32 * stride).floor() as u32;
        indices.push(idx);
    }
    let indices_tensor = Tensor::new(indices, f0.device())?;
    f0.index_select(&indices_tensor, 1).map_err(anyhow::Error::from)
}

fn post_process_rmvpe(f0_probs: &Tensor) -> anyhow::Result<Tensor> {
    let (n_batch, n_frames, n_bins) = f0_probs.dims3()?;
    let argmax = f0_probs.argmax(2)?; 
    let argmax_data = argmax.to_vec2::<u32>()?;
    let mut f0_values = Vec::with_capacity(n_batch * n_frames);
    for batch in argmax_data {
        for bin in batch {
            let cents = (bin as f32) * 20.0 + 1991.303504;
            let f0 = 10.0 * 2.0f32.powf(cents / 1200.0);
            f0_values.push(f0);
        }
    }
    Tensor::from_vec(f0_values, (n_batch, n_frames), f0_probs.device()).map_err(anyhow::Error::from)
}

fn run_component_test() -> anyhow::Result<()> {
    println!("Component test is deprecated. Use test_full binary.");
    Ok(())
}
