use crate::feature_extractor::ContentVec;
use crate::audio_stitching::CrossFadeStitcher;
use crate::rvc_engine::{RvcInferenceEngine, RvcInputs, TractEngine, CandleEngine};
#[cfg(feature = "onnxruntime")]
use crate::rvc_engine::OnnxRuntimeEngine;
use crate::pitch_extractor::Rmvpe;
use candle_core::{Tensor, Device, DType};
use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};
use rubato::{Resampler, FftFixedIn};

#[derive(Debug, Clone, Copy)]
pub enum InferenceBackend {
    Tract,
    Candle,
    #[cfg(feature = "onnxruntime")]
    OnnxRuntime,
}

pub struct RvcPipeline {
    content_vec: ContentVec,
    rmvpe: Rmvpe,
    rvc_engine: Box<dyn RvcInferenceEngine>,
    index_tensor: Option<Tensor>,
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
        index_path: Option<impl AsRef<Path>>,
        device: Device,
        backend: InferenceBackend,
    ) -> Result<Self> {
        let content_vec = ContentVec::new(content_vec_path)
            .context("Failed to load ContentVec model")?;

        let rmvpe = Rmvpe::new(rmvpe_path)
            .context("Failed to load RMVPE model")?;

        let rvc_engine: Box<dyn RvcInferenceEngine> = match backend {
            InferenceBackend::Tract => {
                Box::new(TractEngine::new(rvc_path.as_ref())?)
            }
            InferenceBackend::Candle => {
                Box::new(CandleEngine::new(rvc_path.as_ref(), device.clone())?)
            }
            #[cfg(feature = "onnxruntime")]
            InferenceBackend::OnnxRuntime => {
                Box::new(OnnxRuntimeEngine::new(rvc_path.as_ref())?)
            }
        };

        let index_tensor = if let Some(p) = index_path {
            if p.as_ref().exists() {
                println!("Loading index from {:?}", p.as_ref());
                let tensors = candle_core::safetensors::load(p.as_ref(), &device)?;
                tensors.get("vectors").cloned()
            } else {
                None
            }
        } else {
            None
        };

        let stitcher = CrossFadeStitcher::new(320);

        Ok(Self {
            content_vec,
            rmvpe,
            rvc_engine,
            index_tensor,
            stitcher,
            device,
            model_sr: 40000,
            output_sr: 16000,
        })
    }

    pub fn set_model(&mut self, rvc_path: impl AsRef<Path>, backend: InferenceBackend) -> Result<()> {
        println!("Switching RVC model to {}...", rvc_path.as_ref().display());
        let rvc_engine: Box<dyn RvcInferenceEngine> = match backend {
            InferenceBackend::Tract => {
                Box::new(TractEngine::new(rvc_path.as_ref())?)
            }
            InferenceBackend::Candle => {
                Box::new(CandleEngine::new(rvc_path.as_ref(), self.device.clone())?)
            }
            #[cfg(feature = "onnxruntime")]
            InferenceBackend::OnnxRuntime => {
                Box::new(OnnxRuntimeEngine::new(rvc_path.as_ref())?)
            }
        };
        self.rvc_engine = rvc_engine;
        self.stitcher.reset();
        Ok(())
    }

    pub fn set_index(&mut self, index_path: impl AsRef<Path>) -> Result<()> {
        if index_path.as_ref().exists() {
            println!("Loading index from {:?}", index_path.as_ref());
            let tensors = candle_core::safetensors::load(index_path.as_ref(), &self.device)?;
            self.index_tensor = tensors.get("vectors").cloned();
        }
        Ok(())
    }

    pub fn process(&mut self, audio: &[f32], input_hop_size: usize, pitch_shift: f32, index_rate: f32) -> Result<Vec<f32>> {
        use std::time::Instant;
        let start_total = Instant::now();

        // 1. Extract Features (ContentVec)
        let start = Instant::now();
        let features = self.content_vec.extract(audio)?;
        let (_b, t, d) = (features.shape()[0], features.shape()[1], features.shape()[2]);
        // println!("  [Timing] ContentVec: {:?}", start.elapsed());

        // 2. Extract Pitch (RMVPE)
        let start = Instant::now();
        
        // Use direct waveform forward
        let (f0_vec_raw, _shape) = self.rmvpe.forward(audio, 0.03)?;
        
        // voice-changer RMVPE onnx output is [1, T]. 
        // If the model is E2E (takes waveform, returns pitchf), we use f0_vec_raw directly.
        // We assume shape is [1, T].
        let f0_vec = ndarray::Array1::from_vec(f0_vec_raw);
        
        // println!("  [Timing] RMVPE inference: {:?}, output len: {}", start.elapsed(), f0_vec.len());

        let start = Instant::now();
        let f0_shifted = f0_vec.mapv(|f| if f > 0.0 { f * 2.0f32.powf(pitch_shift / 12.0) } else { 0.0 });

        // 3. Interpolate Features
        let target_len = f0_shifted.len();
        let feats_data = features.as_slice::<f32>()?;
        let feats_interpolated = self.interpolate_features(feats_data, t, d, target_len)?;

        // 4. Prepare inputs
        let mut feats_vec = feats_interpolated;

        // 4.5 Apply Index
        if let Some(index) = &self.index_tensor {
            if index_rate > 0.0 {
                let start = Instant::now();
                let feats_tensor = Tensor::from_vec(feats_vec.clone(), (1, target_len, d), &self.device)?;
                let query = feats_tensor.squeeze(0)?;
                let query_sq = query.sqr()?.sum_keepdim(1)?;
                let index_sq = index.sqr()?.sum_keepdim(1)?.t()?;
                let xy = query.matmul(&index.t()?)?;
                let dist = query_sq.broadcast_add(&index_sq)?
                    .broadcast_sub(&xy.broadcast_mul(&Tensor::new(2.0f32, &self.device)?)?)?;
                let nearest_idx = dist.argmin(1)?;
                let nearest = index.index_select(&nearest_idx, 0)?;
                let ratio = Tensor::new(index_rate, &self.device)?;
                let one_minus_ratio = Tensor::new(1.0 - index_rate, &self.device)?;
                let mixed = query.broadcast_mul(&one_minus_ratio)?
                    .broadcast_add(&nearest.broadcast_mul(&ratio)?)?;
                feats_vec = mixed.flatten_all()?.to_vec1::<f32>()?;
                // println!("  [Timing] Index search: {:?}", start.elapsed());
            }
        }

        let f0_vec_final = f0_shifted.to_vec();
        let pitch_indices: Vec<i64> = f0_vec_final.iter().map(|&f| {
             if f > 0.0 {
                let p = 12.0 * (f / 10.0).log2();
                (p.round() as i64).clamp(1, 255)
            } else {
                0
            }
        }).collect();

        let rvc_inputs = RvcInputs {
            feats: feats_vec,
            pitch: pitch_indices,
            pitchf: f0_vec_final,
            p_len: target_len as i64,
            sid: 0,
            batch_size: 1,
            seq_len: target_len,
            feat_dim: d,
        };

        // 5. Run Inference
        let start = Instant::now();
        let output = self.rvc_engine.infer(rvc_inputs)?;
        let audio_data = output.audio;
        // println!("  [Timing] {} RVC inference: {:?}, output len: {}", self.rvc_engine.backend_name(), start.elapsed(), audio_data.len());

        // 6. Streaming Crop
        let out_len = audio_data.len();
        let keep_len = input_hop_size.min(out_len);
        let cropped = if out_len > keep_len {
            audio_data[out_len - keep_len..].to_vec()
        } else {
            audio_data
        };

        // 7. Resample to output_sr (16k)
        let resampled = if self.model_sr != self.output_sr {
            self.resample(&cropped, self.model_sr, self.output_sr)?
        } else {
            cropped
        };

        // 8. Stitching
        let stitched = self.stitcher.process(resampled);

        println!("  [Timing] TOTAL: {:?}", start_total.elapsed());

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
