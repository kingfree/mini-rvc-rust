use candle_core::{Device, Tensor};
use candle_onnx;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("--- ContentVec Rust Performance Test (Local Fork) ---");

    let device = Device::Cpu;
    let model_path = "pretrain/content_vec_500.onnx";

    if !Path::new(model_path).exists() {
        anyhow::bail!("Model not found: {}", model_path);
    }

    println!("正在加载 ContentVec 模型: {}...", model_path);
    let start_load = Instant::now();
    let model_proto = candle_onnx::read_file(model_path)?;
    println!("模型加载完成，耗时: {:?}", start_load.elapsed());

    let sample_len = 16000; // 1秒音频
    let waveform = Tensor::randn(0f32, 1f32, (1, sample_len), &device)?;

    let mut inputs = HashMap::new();
    inputs.insert("audio".to_string(), waveform);

    println!("开始推理测试 (Local Fork with InstanceNormalization Support)...");
    let start_infer = Instant::now();
    
    match candle_onnx::simple_eval(&model_proto, inputs) {
        Ok(outputs) => {
            let duration = start_infer.elapsed();
            println!("✅ 推理成功！");
            println!("总耗时: {:?}", duration);
            for (name, tensor) in outputs.iter() {
                println!("输出节点 [{}], 形状: {:?}", name, tensor.shape());
            }
        }
        Err(e) => {
            println!("❌ 推理失败: {:?}", e);
        }
    }

    Ok(())
}
