use std::path::Path;
use anyhow::{Context, Result};
use crate::audio_processing::MelSpectrogram;
use ndarray::{Array1, Array2, Axis};

#[cfg(not(feature = "onnxruntime"))]
use tract_onnx::prelude::*;

#[cfg(feature = "onnxruntime")]
use ort::{session::Session, value::Value};

enum RmvpeModelType {
    E2E, // Expects "waveform", returns "pitch" or "f0"
    Raw, // Expects "input" (mel), returns "output" (prob map)
}

pub struct Rmvpe {
    #[cfg(not(feature = "onnxruntime"))]
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    #[cfg(feature = "onnxruntime")]
    session: Session,
    
    model_type: RmvpeModelType,
    mel_extractor: Option<MelSpectrogram>,
}

impl Rmvpe {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        #[cfg(not(feature = "onnxruntime"))]
        {
            let model = tract_onnx::onnx()
                .model_for_path(model_path)?
                .into_optimized()?
                .into_runnable()?;
            // Tract usually loads Raw models in this repo context
            Ok(Self { 
                model,
                model_type: RmvpeModelType::Raw,
                mel_extractor: Some(MelSpectrogram::new(16000, 1024, 160, 1024, 128)),
            })
        }

        #[cfg(feature = "onnxruntime")]
        {
            use ort::session::builder::GraphOptimizationLevel;
            
            // Mirror voice-changer / RVC settings for RMVPE
            // Disable CoreML due to stability issues.
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([
                    ort::execution_providers::CPUExecutionProvider::default().build(),
                ])?
                .commit_from_file(model_path)?;
            
            // Detect model type by input name
            let inputs = session.inputs();
            let first_input_name = inputs[0].name();
            
            let (model_type, mel_extractor) = if first_input_name == "waveform" {
                println!("RMVPE: Detected E2E model (waveform input)");
                (RmvpeModelType::E2E, None)
            } else {
                println!("RMVPE: Detected Raw model ({} input)", first_input_name);
                (RmvpeModelType::Raw, Some(MelSpectrogram::new(16000, 1024, 160, 1024, 128)))
            };

            Ok(Self { session, model_type, mel_extractor })
        }
    }

    pub fn forward(&mut self, audio: &[f32], threshold: f32) -> Result<(Vec<f32>, Vec<usize>)> {
        match self.model_type {
            RmvpeModelType::E2E => self.forward_e2e(audio, threshold),
            RmvpeModelType::Raw => self.forward_raw(audio, threshold),
        }
    }

    #[cfg(feature = "onnxruntime")]
    fn forward_e2e(&mut self, audio: &[f32], threshold: f32) -> Result<(Vec<f32>, Vec<usize>)> {
        let len = audio.len();
        let shape = [1, len as i64];
        
        let audio_val = Value::from_array((shape, audio.to_vec()))?;
        let thresh_val = Value::from_array(([1], vec![threshold]))?;
        
        let outputs = self.session.run(ort::inputs![
            "waveform" => audio_val,
            "threshold" => thresh_val
        ])?;
        
        let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_vec = out_shape.iter().map(|&x| x as usize).collect();
        
        Ok((out_data.to_vec(), shape_vec))
    }

    #[cfg(feature = "onnxruntime")]
    fn forward_raw(&mut self, audio: &[f32], _threshold: f32) -> Result<(Vec<f32>, Vec<usize>)> {
        // 1. Extract Mel
        let mel_extractor = self.mel_extractor.as_ref().context("Mel extractor not initialized")?;
        let mel_spec = mel_extractor.forward(audio)?;
        let (n_mels, n_frames) = mel_spec.dim();
        
        // 2. Prepare Input [1, 128, T]
        let mel_flat: Vec<f32> = mel_spec.iter().cloned().collect();
        let shape = [1, n_mels as i64, n_frames as i64];
        let input_val = Value::from_array((shape, mel_flat))?;

        // 3. Run Inference
        let (probs_vec, shape_vec) = {
            let outputs = self.session.run(ort::inputs![input_val])?;
            let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;
            (out_data.to_vec(), out_shape.to_vec())
        };
        
        // 4. Post Process
        // Output shape from ONNX verify: [1, T, 360]
        let probs = Array1::from_vec(probs_vec)
            .into_shape_with_order((shape_vec[0] as usize, shape_vec[1] as usize, shape_vec[2] as usize))?;
        
        // probs: [1, T, 360]
        let f0 = Self::post_process_probs(probs.view())?;
        
        Ok((f0.to_vec(), f0.shape().to_vec()))
    }

    fn post_process_probs(probs: ndarray::ArrayView3<f32>) -> Result<Array1<f32>> {
        // probs: [1, T, 360]
        let (_b, n_frames, n_classes) = probs.dim();
        let mut f0_values = Vec::with_capacity(n_frames);

        for t in 0..n_frames {
            let mut max_val = -f32::INFINITY;
            let mut max_idx = 0;
            for c in 0..n_classes {
                let val = probs[[0, t, c]];
                if val > max_val {
                    max_val = val;
                    max_idx = c;
                }
            }
            
            if max_val < 0.03 {
                f0_values.push(0.0);
            } else {
                let cents = (max_idx as f32) * 20.0 + 1997.3794084;
                let f0 = 10.0 * 2.0f32.powf(cents / 1200.0);
                f0_values.push(f0);
            }
        }
        
        Ok(Array1::from(f0_values))
    }

    #[cfg(not(feature = "onnxruntime"))]
    fn forward_e2e(&mut self, _audio: &[f32], _threshold: f32) -> Result<(Vec<f32>, Vec<usize>)> {
        panic!("E2E model not supported in Tract backend");
    }

    #[cfg(not(feature = "onnxruntime"))]
    fn forward_raw(&mut self, audio: &[f32], _threshold: f32) -> Result<(Vec<f32>, Vec<usize>)> {
        // Tract implementation
        let mel_extractor = self.mel_extractor.as_ref().unwrap();
        let mel_spec = mel_extractor.forward(audio)?;
        let (n_mels, n_frames) = mel_spec.dim();
        
        let mel_flat: Vec<f32> = mel_spec.iter().cloned().collect();
        let input = Tensor::from_shape(&[1, n_mels, n_frames], &mel_flat)?;
        
        let result = self.model.run(tvec!(input.into()))?;
        let output = result[0].to_array_view::<f32>()?;
        
        // Post process tract output
        let probs = output.into_shape_with_order((1, 360, n_frames))?;
        let f0 = Self::post_process_probs(probs.view())?;
        
        Ok((f0.to_vec(), f0.shape().to_vec()))
    }
}