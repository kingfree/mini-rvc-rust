mod audio_processing;
mod feature_extractor;
mod pitch_extractor;

use std::time::Instant;
use tract_onnx::prelude::*;
use candle_core::{Device, Tensor};
use candle_onnx;

fn main() -> anyhow::Result<()> {
    println!("--- RVC Rust Inference Components Test ---");

    let device = Device::Cpu;

    // 1. 加载音频
    let wav_path = "assets/test.wav";
    if !std::path::Path::new(wav_path).exists() {
        println!("错误: 未找到音频文件 {}。", wav_path);
        return Ok(());
    }
    
    let (waveform, sr) = feature_extractor::load_wav(wav_path)?;
    println!("音频载入成功，采样率: {}, 总采样点: {}", sr, waveform.len());

    // 预处理: 重采样到 16kHz
    let waveform_16k = audio_processing::resample_to_16k(&waveform, sr);
    println!("已重采样至 16kHz, 采样点数: {}", waveform_16k.len());

    // 2. ContentVec 特征提取 (Tract)
    let hubert_path = "pretrain/content_vec_500.onnx";
    if std::path::Path::new(hubert_path).exists() {
        println!("正在加载 ContentVec (Tract): {}...", hubert_path);
        let extractor = feature_extractor::ContentVec::new(hubert_path)?;
        
        // 取前 1 秒测试
        let test_len = 16000.min(waveform_16k.len());
        let test_samples = &waveform_16k[..test_len];

        let start = Instant::now();
        let features = extractor.extract(test_samples)?;
        println!("✅ ContentVec 提取成功！耗时: {:?}, 形状: {:?}", start.elapsed(), features.shape());
    }

    // 3. RMVPE 音高提取
    let rmvpe_path = "pretrain/rmvpe.onnx";
    if std::path::Path::new(rmvpe_path).exists() {
        println!("正在加载 RMVPE 模型 (Candle)...");
        let model_proto = candle_onnx::read_file(rmvpe_path)?;
        
        let mel_extractor = audio_processing::MelSpectrogram::new(16000, 1024, 160, 1024, 128);
        println!("计算 Mel 频谱...");
        let mel = mel_extractor.forward(&waveform_16k)?;
        
        // 转换 ndarray -> candle Tensor [1, 128, T]
        let (n_mels, n_frames) = mel.dim();
        let mel_flat: Vec<f32> = mel.iter().cloned().collect();
        let mel_tensor = Tensor::from_vec(mel_flat, (1, n_mels, n_frames), &device)?;
        
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("input".to_string(), mel_tensor);

        println!("执行 RMVPE 推理...");
        let start_rmvpe = Instant::now();
        match candle_onnx::simple_eval(&model_proto, inputs) {
            Ok(outputs) => {
                println!("✅ RMVPE 推理成功！耗时: {:?}", start_rmvpe.elapsed());
                if let Some(f0) = outputs.get("f0") {
                    println!("   F0 形状: {:?}", f0.shape());
                }
            }
            Err(e) => {
                println!("❌ RMVPE 推理失败: {:?}", e);
            }
        }
    }

    println!("--- 组件测试完成 ---");
    Ok(())
}
