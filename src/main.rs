use candle_core::{Device, Tensor};
use candle_onnx;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("--- RVC Rust Performance Test (Anon v2 - Local Fork) ---");

    let device = Device::Cpu;
    let model_path = "models/Anon_v2_merged.onnx";

    if !Path::new(model_path).exists() {
        anyhow::bail!("Model not found: {}", model_path);
    }

    println!("正在加载模型: {}...", model_path);
    let start_load = Instant::now();
    let model_proto = candle_onnx::read_file(model_path)?;
    println!("模型加载完成，耗时: {:?}", start_load.elapsed());

    // 导出时硬编码了 64 帧
    let num_frames = 64; 
    println!("测试输入帧数: {}", num_frames);

    let feats = Tensor::randn(0f32, 1f32, (1, num_frames, 768), &device)?;
    let p_len = Tensor::from_slice(&[num_frames as i64], (1,), &device)?;
    let pitch = Tensor::zeros((1, num_frames), candle_core::DType::I64, &device)?;
    let pitchf = Tensor::zeros((1, num_frames), candle_core::DType::F32, &device)?;
    let sid = Tensor::from_slice(&[0i64], (1,), &device)?;

    let mut inputs = HashMap::new();
    inputs.insert("feats".to_string(), feats);
    inputs.insert("p_len".to_string(), p_len);
    inputs.insert("pitch".to_string(), pitch);
    inputs.insert("pitchf".to_string(), pitchf);
    inputs.insert("sid".to_string(), sid);

    println!("开始推理测试 (Local Fork with Pad Support)...");
    let start_infer = Instant::now();
    
    match candle_onnx::simple_eval(&model_proto, inputs) {
        Ok(outputs) => {
            let duration = start_infer.elapsed();
            println!("✅ 推理成功！");
            println!("总耗时: {:?}", duration);
            if let Some(audio) = outputs.get("audio") {
                println!("输出音频形状: {:?}", audio.shape());
            }
        }
        Err(e) => {
            println!("❌ 推理失败: {:?}", e);
        }
    }

    Ok(())
}
