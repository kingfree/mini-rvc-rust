use crate::feature_extractor::ContentVec;
use crate::audio_processing::{self, MelSpectrogram};
use crate::audio_stitching::CrossFadeStitcher;
use candle_core::{Tensor, Device, DType};
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;
use anyhow::{Context, Result};
use rubato::{Resampler, FftFixedIn};

pub struct RvcPipeline {
    content_vec: ContentVec,
    // RMVPE using Tract
    rmvpe: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    mel_extractor: MelSpectrogram,
    // RVC Model (Candle)
    rvc_model_proto: candle_onnx::onnx::ModelProto,
    
    stitcher: CrossFadeStitcher,
    device: Device,
    
    model_sr: u32,
    output_sr: u32,
}

impl RvcPipeline {
    pub fn new(
        content_vec_path: impl AsRef<Path>,
        rmvpe_path: impl AsRef<Path>,
        rvc_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        // 1. Load ContentVec
        let content_vec = ContentVec::new(content_vec_path)
            .context("Failed to load ContentVec model")?;

        // 2. Load RMVPE (Tract)
        // RMVPE expects 128 mel bins.
        // We use concrete input length (128 frames) because tract symbolic inference
        // fails on this model's complex U-Net structure (output mismatch) if dynamic.
        let rmvpe = tract_onnx::onnx()
            .model_for_path(rmvpe_path.as_ref())?
            .with_input_fact(0, f32::fact([1, 128, 128]).into())?
            .into_optimized()?
            .into_runnable()?;
            
        // Use 128 mels!
        let mel_extractor = MelSpectrogram::new(16000, 1024, 160, 1024, 128);

        // 3. Load RVC Model (Candle)
        let rvc_model_proto = candle_onnx::read_file(rvc_path.as_ref())
            .context("Failed to load RVC model")?;

        // 4. Stitcher
        // Fade length: e.g. 20ms at 16kHz = 320 samples.
        let stitcher = CrossFadeStitcher::new(320);

        Ok(Self {
            content_vec,
            rmvpe,
            mel_extractor,
            rvc_model_proto,
            stitcher,
            device,
            model_sr: 40000, // Detected from Yukina_v2
            output_sr: 16000, // Match frontend for now
        })
    }

    pub fn set_model(&mut self, rvc_path: impl AsRef<Path>) -> Result<()> {
        let rvc_model_proto = candle_onnx::read_file(rvc_path.as_ref())
            .context("Failed to load new RVC model")?;
        self.rvc_model_proto = rvc_model_proto;
        self.stitcher.reset(); 
        Ok(())
    }

    pub fn process(&mut self, audio: &[f32], input_hop_size: usize, pitch_shift: f32) -> Result<Vec<f32>> {
        // 1. Extract Features (ContentVec)
        let features = self.content_vec.extract(audio)?;
        let (_b, t, d) = (features.shape()[0], features.shape()[1], features.shape()[2]);
        
        // 2. Extract Pitch (RMVPE)
        let mel = self.mel_extractor.forward(audio)?;
        let (n_mels, n_frames) = mel.dim();
        
        let mel_flat: Vec<f32> = mel.iter().cloned().collect();
        let mel_tract = tract_onnx::prelude::Tensor::from_shape(&[1, n_mels, n_frames], &mel_flat)?;
        
        let rmvpe_out = self.rmvpe.run(tvec!(mel_tract.into()))?;
        let f0_probs = rmvpe_out[0].to_array_view::<f32>()?;
        let f0 = self.post_process_f0(f0_probs)?;
        let f0_shifted = f0.mapv(|f| if f > 0.0 { f * 2.0f32.powf(pitch_shift / 12.0) } else { 0.0 });

        // 3. Interpolate Features
        let target_len = f0_shifted.len();
        let feats_data = features.as_slice::<f32>()?;
        let feats_interpolated = self.interpolate_features(feats_data, t, d, target_len)?;
        
        // 4. Prepare inputs
        let feats_tensor = Tensor::from_vec(feats_interpolated, (1, target_len, d), &self.device)?;
        let f0_vec = f0_shifted.to_vec();
        let pitch_indices: Vec<i64> = f0_vec.iter().map(|&f| {
             if f > 0.0 {
                let p = 12.0 * (f / 10.0).log2();
                (p.round() as i64).clamp(1, 255)
            } else {
                0
            }
        }).collect();
        
        let pitch = Tensor::from_slice(&pitch_indices, (1, target_len), &self.device)?;
        let pitchf = Tensor::from_slice(&f0_vec, (1, target_len), &self.device)?;
        let p_len = Tensor::from_slice(&[target_len as i64], (1,), &self.device)?;
        let sid = Tensor::from_slice(&[0i64], (1,), &self.device)?;
        
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("feats".to_string(), feats_tensor);
        inputs.insert("p_len".to_string(), p_len);
        inputs.insert("pitch".to_string(), pitch);
        inputs.insert("pitchf".to_string(), pitchf);
        inputs.insert("sid".to_string(), sid);
        
        // 5. Run Inference
        let outputs = candle_onnx::simple_eval(&self.rvc_model_proto, inputs)?;
        let audio_out = outputs.get("audio").context("Model output 'audio' not found")?;
        let mut audio_data = audio_out.flatten_all()?.to_vec1::<f32>()?;

        // 6. Resample FIRST to match target rate (16k) so we can crop by samples reliably
        let resampled = if self.model_sr != self.output_sr {
            self.resample(&audio_data, self.model_sr, self.output_sr)?
        } else {
            audio_data
        };

        // 7. Streaming Crop
        // input_hop_size is in 16k samples. resampled is 16k.
        // We want to keep the LAST input_hop_size samples.
        let out_len = resampled.len();
        let keep_len = input_hop_size.min(out_len);
        let cropped = if out_len > keep_len {
            resampled[out_len - keep_len..].to_vec()
        } else {
            resampled
        };
        
        // 8. Stitching
        let stitched = self.stitcher.process(cropped);
        
        Ok(stitched)
    }

    fn interpolate_features(&self, data: &[f32], old_t: usize, d: usize, new_t: usize) -> Result<Vec<f32>> {
        if old_t == new_t {
            return Ok(data.to_vec());
        }
        let mut out = Vec::with_capacity(new_t * d);
        let ratio = (old_t - 1) as f32 / (new_t - 1) as f32;
        for i in 0..new_t {
            let pos = i as f32 * ratio;
            let idx = pos.floor() as usize;
            let frac = pos - idx as f32;
            let next_idx = (idx + 1).min(old_t - 1);
            
            for j in 0..d {
                let v1 = data[idx * d + j];
                let v2 = data[next_idx * d + j];
                out.push(v1 * (1.0 - frac) + v2 * frac);
            }
        }
        Ok(out)
    }
    
    fn post_process_f0(&self, f0_probs: ndarray::ArrayViewD<f32>) -> Result<ndarray::Array1<f32>> {
         // f0_probs: [1, n_frames, 360]
        let shape = f0_probs.shape();
        let n_frames = shape[1];
        
        // Argmax
        let mut f0_values = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let mut max_val = -f32::INFINITY;
            let mut max_idx = 0;
            for j in 0..360 {
                let val = f0_probs[[0, i, j]];
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            
            let cents = (max_idx as f32) * 20.0 + 1991.303504;
            let f0 = 10.0 * 2.0f32.powf(cents / 1200.0);
             if max_val < 0.05 { 
                f0_values.push(0.0);
            } else {
                f0_values.push(f0);
            }
        }
        
        Ok(ndarray::Array1::from(f0_values))
    }

    fn resample(&self, input: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
        let mut resampler = FftFixedIn::<f32>::new(
            from as usize,
            to as usize,
            input.len(),
            1,
            1,
        )?;
        let out = resampler.process(&[input], None)?;
        Ok(out[0].clone())
    }
}