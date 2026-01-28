use tract_onnx::prelude::*;
use hound;
use std::path::Path;
use std::sync::Arc;

pub struct ContentVec {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl ContentVec {
    pub fn new(model_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(0, f32::fact(&[1, 16000]).into())? 
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    pub fn extract(&self, waveform: &[f32]) -> anyhow::Result<Arc<Tensor>> {
        let len = waveform.len();
        let input = Tensor::from_shape(&[1, len], waveform)?;
        let mut outputs = self.model.run(tvec!(input.into()))?;
        let result = outputs.remove(0).into_tensor();
        Ok(Arc::new(result))
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
