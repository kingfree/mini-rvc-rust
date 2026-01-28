use tract_onnx::prelude::*;
use std::path::Path;
use std::sync::Arc;

pub struct RMVPE {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl RMVPE {
    pub fn new(model_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    pub fn estimate(&self, waveform: &[f32]) -> anyhow::Result<Arc<Tensor>> {
        let len = waveform.len();
        // RMVPE usually expects [1, T] 16kHz
        let input = Tensor::from_shape(&[1, len], waveform)?;
        let mut outputs = self.model.run(tvec!(input.into()))?;
        let result = outputs.remove(0).into_tensor();
        Ok(Arc::new(result))
    }
}
