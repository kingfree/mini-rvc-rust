mod feature_extractor;
mod pitch_extractor;

use std::time::Instant;
use tract_onnx::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("--- RVC Rust Inference Components Test ---");

    // 1. 加载音频
    let wav_path = "assets/test.wav";
    if !std::path::Path::new(wav_path).exists() {
        println!("错误: 未找到音频文件 {}。", wav_path);
        return Ok(());
    }
    
    let (waveform, sr) = feature_extractor::load_wav(wav_path)?;
    println!("音频载入成功，采样率: {}, 总采样点: {}", sr, waveform.len());

    // 预处理: 取前 1 秒
    let test_samples = if waveform.len() > 16000 {
        &waveform[..16000]
    } else {
        &waveform
    };

    // 2. ContentVec 特征提取 (通过 Tract 实现纯 Rust 推理)
    let hubert_path = "pretrain/content_vec_500.onnx";
    if std::path::Path::new(hubert_path).exists() {
        println!("正在通过 Tract 加载 ContentVec (Hubert)...");
        let extractor = feature_extractor::ContentVec::new(hubert_path)?;
        let start = Instant::now();
        let features = extractor.extract(test_samples)?;
        println!("✅ ContentVec 提取成功！耗时: {:?}, 形状: {:?}", start.elapsed(), features.shape());
        println!("   提示: RVC 实时转换的核心特征提取已跑通。");
    }

    // 3. RMVPE 与 角色模型 状态说明
    println!("\n--- 待解决的技术挑战 ---");
    println!("1. RMVPE: 模型期待 [1, 128, T] 输入 (Mel Spectrogram)，需在 Rust 中实现预处理。");
    println!("2. 角色模型: Candle 暂不支持 Pad constant 算子，Tract 暂不支持 RandomNormalLike。");
    println!("   后续计划: 手动实现缺失算子或优化 ONNX 模型图。");

    println!("\n--- 组件测试完成 ---");
    Ok(())
}
