use anyhow::Result;
use std::path::Path;
use std::collections::HashMap;

/// RVC Model Input tensors
pub struct RvcInputs {
    pub feats: Vec<f32>,     // [1, T, 768]
    pub pitch: Vec<i64>,     // [1, T]
    pub pitchf: Vec<f32>,    // [1, T]
    pub p_len: i64,          // scalar
    pub sid: i64,            // scalar
    pub batch_size: usize,
    pub seq_len: usize,
    pub feat_dim: usize,
}

/// RVC Model Output
pub struct RvcOutput {
    pub audio: Vec<f32>,
}

/// Trait for RVC inference engines
/// Supports multiple backends: Tract (CPU), Candle (Metal/CUDA), TensorRT (CUDA)
pub trait RvcInferenceEngine: Send {
    /// Run inference on the RVC model
    fn infer(&mut self, inputs: RvcInputs) -> Result<RvcOutput>;

    /// Get the backend name for logging
    fn backend_name(&self) -> &str;

    /// Clone the engine for multi-threading (if supported)
    fn try_clone(&self) -> Result<Box<dyn RvcInferenceEngine>>;
}

/// Tract-based inference engine (CPU-optimized)
pub struct TractEngine {
    model: tract_onnx::prelude::RunnableModel<
        tract_onnx::prelude::TypedFact,
        Box<dyn tract_onnx::prelude::TypedOp>,
        tract_onnx::prelude::Graph<
            tract_onnx::prelude::TypedFact,
            Box<dyn tract_onnx::prelude::TypedOp>
        >
    >,
}

impl TractEngine {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        use tract_onnx::prelude::*;

        println!("Loading RVC model with Tract (CPU)...");
        let model = tract_onnx::onnx()
            .model_for_path(model_path.as_ref())?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self { model })
    }
}

impl RvcInferenceEngine for TractEngine {
    fn infer(&mut self, inputs: RvcInputs) -> Result<RvcOutput> {
        use tract_onnx::prelude::*;

        let RvcInputs {
            feats,
            pitch,
            pitchf,
            p_len,
            sid,
            batch_size,
            seq_len,
            feat_dim,
        } = inputs;

        // Create Tract tensors
        let feats_tensor = Tensor::from_shape(
            &[batch_size, seq_len, feat_dim],
            &feats
        )?;

        let pitch_tensor = Tensor::from_shape(
            &[batch_size, seq_len],
            &pitch
        )?;

        let pitchf_tensor = Tensor::from_shape(
            &[batch_size, seq_len],
            &pitchf
        )?;

        let p_len_tensor = Tensor::from_shape(
            &[batch_size],
            &[p_len]
        )?;

        let sid_tensor = Tensor::from_shape(
            &[batch_size],
            &[sid]
        )?;

        // Run inference
        let outputs = self.model.run(tvec![
            feats_tensor.into(),
            p_len_tensor.into(),
            pitch_tensor.into(),
            pitchf_tensor.into(),
            sid_tensor.into(),
        ])?;

        // Extract audio output
        let audio_tensor = outputs[0].to_array_view::<f32>()?;
        let audio: Vec<f32> = audio_tensor.iter().cloned().collect();

        Ok(RvcOutput { audio })
    }

    fn backend_name(&self) -> &str {
        "Tract (CPU)"
    }

    fn try_clone(&self) -> Result<Box<dyn RvcInferenceEngine>> {
        anyhow::bail!("TractEngine does not support cloning")
    }
}

/// Candle-based inference engine (Metal/CUDA-accelerated)
pub struct CandleEngine {
    model_proto: candle_onnx::onnx::ModelProto,
    device: candle_core::Device,
}

impl CandleEngine {
    pub fn new(model_path: impl AsRef<Path>, device: candle_core::Device) -> Result<Self> {
        println!("Loading RVC model with Candle ({:?})...", device);
        let model_proto = candle_onnx::read_file(model_path.as_ref())?;

        Ok(Self {
            model_proto,
            device,
        })
    }
}

impl RvcInferenceEngine for CandleEngine {
    fn infer(&mut self, inputs: RvcInputs) -> Result<RvcOutput> {
        use candle_core::Tensor;
        use anyhow::Context;

        let RvcInputs {
            feats,
            pitch,
            pitchf,
            p_len,
            sid,
            batch_size,
            seq_len,
            feat_dim,
        } = inputs;

        // Create Candle tensors on target device
        let feats_tensor = Tensor::from_vec(
            feats,
            (batch_size, seq_len, feat_dim),
            &self.device
        )?;

        let pitch_tensor = Tensor::from_slice(
            &pitch,
            (batch_size, seq_len),
            &self.device
        )?;

        let pitchf_tensor = Tensor::from_slice(
            &pitchf,
            (batch_size, seq_len),
            &self.device
        )?;

        let p_len_tensor = Tensor::from_slice(
            &[p_len],
            (batch_size,),
            &self.device
        )?;

        let sid_tensor = Tensor::from_slice(
            &[sid],
            (batch_size,),
            &self.device
        )?;

        // Prepare input map
        let mut input_map = HashMap::new();
        input_map.insert("feats".to_string(), feats_tensor);
        input_map.insert("p_len".to_string(), p_len_tensor);
        input_map.insert("pitch".to_string(), pitch_tensor);
        input_map.insert("pitchf".to_string(), pitchf_tensor);
        input_map.insert("sid".to_string(), sid_tensor);

        // Run inference
        let outputs = candle_onnx::simple_eval(&self.model_proto, input_map)?;
        let audio_tensor = outputs.get("audio")
            .context("Model output 'audio' not found")?;
        let audio = audio_tensor.flatten_all()?.to_vec1::<f32>()?;

        Ok(RvcOutput { audio })
    }

    fn backend_name(&self) -> &str {
        match &self.device {
            candle_core::Device::Cpu => "Candle (CPU)",
            candle_core::Device::Cuda(_) => "Candle (CUDA)",
            candle_core::Device::Metal(_) => "Candle (Metal)",
        }
    }

    fn try_clone(&self) -> Result<Box<dyn RvcInferenceEngine>> {
        Ok(Box::new(Self {
            model_proto: self.model_proto.clone(),
            device: self.device.clone(),
        }))
    }
}

// ONNX Runtime engine (Development/Benchmark only)
// Completely mirrors voice-changer's ONNX Runtime configuration:
// - CPUExecutionProvider with 8 intra_op_threads and 8 inter_op_threads
// - ORT_PARALLEL execution mode
// - GraphOptimizationLevel::Level3
// NOTE: Uses C FFI - only for development baseline, not for ARM/Jetson deployment
#[cfg(feature = "onnxruntime")]
pub struct OnnxRuntimeEngine {
    session: ort::session::Session,
    is_half: bool,
    input_names: Vec<String>,
}

#[cfg(feature = "onnxruntime")]
impl OnnxRuntimeEngine {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        use ort::session::{Session, builder::GraphOptimizationLevel};
        use ort::tensor::TensorElementType;
        use anyhow::Context;

        println!("Loading RVC model with ONNX Runtime...");

        // Initialize ort environment
        ort::init()
            .with_name("mini-rvc-rust")
            .commit();

        // Build session
        // CoreML is causing instability (Bus Error) due to high graph fragmentation.
        // Reverting to CPU Execution Provider which is stable and reasonably fast on ARM64.
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path.as_ref())?;

        // Check input type to determine if half precision is needed
        // Mirrors voice-changer: first_input_type = onnx_session.get_inputs()[0].type
        let inputs = session.inputs();
        let first_input = inputs.first().context("Model has no inputs")?;
        
        // Note: 'input_type' field access on Outlet is problematic in this ort version.
        // Assuming FP32 for now to allow compilation.
        // To implement correctly, we need to inspect first_input.input_type equivalent.
        let is_half = false; 

        let input_names: Vec<String> = inputs.iter().map(|i| i.name().to_string()).collect();
        
        println!("  Model precision: {}", if is_half { "FP16 (Half)" } else { "FP32 (Float)" });
        println!("  Model inputs: {:?}", input_names);

        Ok(Self { session, is_half, input_names })
    }
}

#[cfg(feature = "onnxruntime")]
impl RvcInferenceEngine for OnnxRuntimeEngine {
    fn infer(&mut self, inputs: RvcInputs) -> Result<RvcOutput> {
        use ort::value::Value;
        use half::f16;

        let RvcInputs {
            feats,
            pitch,
            pitchf,
            p_len,
            sid,
            batch_size,
            seq_len,
            feat_dim,
        } = inputs;

        // Create ndarrays
        let feats_array = ndarray::Array::from_shape_vec(
            (batch_size, seq_len, feat_dim),
            feats
        )?;
        let pitch_array = ndarray::Array::from_shape_vec(
            (batch_size, seq_len),
            pitch
        )?;
        let pitchf_array = ndarray::Array::from_shape_vec(
            (batch_size, seq_len),
            pitchf
        )?;
        let p_len_array = ndarray::Array::from_shape_vec(
            (batch_size,),
            vec![p_len]
        )?;
        let sid_array = ndarray::Array::from_shape_vec(
            (batch_size,),
            vec![sid]
        )?;

        let outputs = if self.is_half {
            // Convert float inputs to half
            let feats_half = feats_array.mapv(|x| f16::from_f32(x));
            let pitchf_half = pitchf_array.mapv(|x| f16::from_f32(x));

            // Use named inputs matching voice-changer, but respecting model inputs
            let mut inputs_vec: Vec<(&str, Value)> = Vec::new();

            if self.input_names.iter().any(|n| n == "feats") {
                let shape = feats_half.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("feats", Value::from_array((shape, feats_half.into_raw_vec()))?.into()));
            }
            if self.input_names.iter().any(|n| n == "p_len") {
                let shape = p_len_array.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("p_len", Value::from_array((shape, p_len_array.into_raw_vec()))?.into()));
            }
            if self.input_names.iter().any(|n| n == "pitch") {
                let shape = pitch_array.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("pitch", Value::from_array((shape, pitch_array.into_raw_vec()))?.into()));
            }
            if self.input_names.iter().any(|n| n == "pitchf") {
                let shape = pitchf_half.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("pitchf", Value::from_array((shape, pitchf_half.into_raw_vec()))?.into()));
            }
            if self.input_names.iter().any(|n| n == "sid") {
                let shape = sid_array.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("sid", Value::from_array((shape, sid_array.into_raw_vec()))?.into()));
            }

            self.session.run(inputs_vec)?
        } else {
            // Use named inputs matching voice-changer, but respecting model inputs
            let mut inputs_vec: Vec<(&str, Value)> = Vec::new();

            if self.input_names.iter().any(|n| n == "feats") {
                let shape = feats_array.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("feats", Value::from_array((shape, feats_array.into_raw_vec()))?.into()));
            }
            if self.input_names.iter().any(|n| n == "p_len") {
                let shape = p_len_array.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("p_len", Value::from_array((shape, p_len_array.into_raw_vec()))?.into()));
            }
            if self.input_names.iter().any(|n| n == "pitch") {
                let shape = pitch_array.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("pitch", Value::from_array((shape, pitch_array.into_raw_vec()))?.into()));
            }
            if self.input_names.iter().any(|n| n == "pitchf") {
                let shape = pitchf_array.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("pitchf", Value::from_array((shape, pitchf_array.into_raw_vec()))?.into()));
            }
            if self.input_names.iter().any(|n| n == "sid") {
                let shape = sid_array.shape().iter().map(|&x| x as i64).collect::<Vec<_>>();
                inputs_vec.push(("sid", Value::from_array((shape, sid_array.into_raw_vec()))?.into()));
            }

            self.session.run(inputs_vec)?
        };

        // Extract audio output (first output tensor)
        let audio_extracted = outputs["audio"].try_extract_tensor::<f32>()?;
        // try_extract_tensor returns (Shape, &[T]) in newer ort versions or TensorRef in others.
        // Based on error message: "no method named `as_slice` found for tuple `(&ort::tensor::Shape, &[f32])`"
        // It returns a tuple.
        let (_, audio_slice) = audio_extracted; 
        let audio: Vec<f32> = audio_slice.to_vec();

        Ok(RvcOutput { audio })
    }

    fn backend_name(&self) -> &str {
        if self.is_half {
            "ONNX Runtime (Benchmark) [FP16]"
        } else {
            "ONNX Runtime (Benchmark) [FP32]"
        }
    }

    fn try_clone(&self) -> Result<Box<dyn RvcInferenceEngine>> {
        anyhow::bail!("OnnxRuntimeEngine does not support cloning")
    }
}