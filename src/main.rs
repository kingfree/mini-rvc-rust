mod feature_extractor;

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("--- RVC Feature Extraction Test (ContentVec) ---");

    // 1. 加载音频
    let wav_path = "assets/test.wav";
    if !std::path::Path::new(wav_path).exists() {
        println!("警告: 未找到音频文件 {}, 将跳过实际提取测试。", wav_path);
        return Ok(());
    }
    
    let (waveform, sr) = feature_extractor::load_wav(wav_path)?;
    println!("音频载入成功，采样率: {}, 采样点数: {}", sr, waveform.len());

    // 2. 加载模型
    let model_path = "pretrain/content_vec_500.onnx";
    if !std::path::Path::new(model_path).exists() {
        println!("警告: 未找到模型文件 {}, 将跳过实际提取测试。", model_path);
        return Ok(());
    }

    println!("正在通过 Tract 加载 ContentVec 模型: {}...", model_path);
    let start_load = Instant::now();
    let extractor = feature_extractor::ContentVec::new(model_path)?;
    println!("模型加载成功，耗时: {:?}", start_load.elapsed());

    // 3. 执行提取
    println!("开始提取特征...");
    let start_extract = Instant::now();
    // 取前 1 秒进行测试
    let test_samples = if waveform.len() > 16000 {
        &waveform[..16000]
    } else {
        &waveform
    };
    
    let features = extractor.extract(test_samples)?;
    println!("特征提取成功！耗时: {:?}", start_extract.elapsed());
    println!("特征形状: {:?}", features.shape());

    Ok(())
}
