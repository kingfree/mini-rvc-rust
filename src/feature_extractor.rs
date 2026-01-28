use tract_onnx::prelude::*;
use hound;
use std::path::Path;
use std::sync::Arc;

pub struct ContentVec {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl ContentVec {
    pub fn new(model_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        // 不强制指定 shape，让 tract 自动处理动态维度
        let model_unoptimized = tract_onnx::onnx()
            .model_for_path(model_path.as_ref())?;
        
        println!("ContentVec Model Inputs:");
        for i in 0..model_unoptimized.inputs.len() {
            let id = model_unoptimized.inputs[i];
            let fact = model_unoptimized.input_fact(i)?;
            println!("  Input {}: id={:?}, name={:?}, fact={:?}", i, id, model_unoptimized.node(id.node).name, fact);
        }
        
        println!("ContentVec Model Outputs:");
        for i in 0..model_unoptimized.outputs.len() {
            let id = model_unoptimized.outputs[i];
            let fact = model_unoptimized.output_fact(i)?;
            println!("  Output {}: id={:?}, name={:?}, fact={:?}", i, id, model_unoptimized.node(id.node).name, fact);
        }

        let model = model_unoptimized
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    pub fn extract(&self, waveform: &[f32]) -> anyhow::Result<Arc<Tensor>> {
        let len = waveform.len();
        // 输入形状应该是 [1, T]
        let input = Tensor::from_shape(&[1, len], waveform)?;
        let mut outputs = self.model.run(tvec!(input.into()))?;
        
        // 我们需要 768 维的特征 (RVC v2)
        // 在 content_vec_500.onnx 中，Output 1 和 2 是 768 维的
        if outputs.len() > 1 {
            let result = outputs.remove(1).into_tensor();
            Ok(Arc::new(result))
        } else {
            let result = outputs.remove(0).into_tensor();
            Ok(Arc::new(result))
        }
    }
}

pub fn load_wav(path: impl AsRef<Path>) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    
    Ok((samples, sample_rate))
}

pub fn load_wav_memory(data: &[u8]) -> anyhow::Result<(Vec<f32>, u32)> {
    let cursor = std::io::Cursor::new(data);
    let mut reader = hound::WavReader::new(cursor)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    
    Ok((samples, sample_rate))
}
