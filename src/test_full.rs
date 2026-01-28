use std::path::Path;
use std::fs::File;
use hound;
use candle_core::Device;
use crate::realtime_pipeline::RvcPipeline;

mod audio_processing;
mod feature_extractor;
mod pitch_extractor;
mod audio_stitching;
mod realtime_pipeline;
mod ring_buffer;
mod rvc_engine;

use realtime_pipeline::InferenceBackend;

fn main() -> anyhow::Result<()> {
    println!("--- Full Pipeline Test ---");

    let content_vec_path = "pretrain/content_vec_500.onnx";
    let rmvpe_path = "pretrain/rmvpe.onnx";
    let rvc_path = "models/Yukina_v2_merged.onnx";
    let index_path: Option<&str> = None; // Disable index for now

    // Test Candle CPU
    println!("\n=== Testing Candle CPU Backend ===");
    let device_cpu = Device::Cpu;
    let backend_cpu = InferenceBackend::Candle;
    let mut pipeline = RvcPipeline::new(content_vec_path, rmvpe_path, rvc_path, index_path, device_cpu, backend_cpu)?;

    let input_path = "assets/test.wav";
    let output_path = "assets/output.wav";

    println!("Loading {}", input_path);
    let mut reader = hound::WavReader::open(input_path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // Resample to 16k if needed (simple decimation for test if 48k/44.1k)
    // The pipeline expects chunks. Let's feed it in 1s chunks (16000 samples).
    
    // Simple resample to 16k for this test
    // Assuming input is > 16k.
    let samples_16k = audio_processing::resample_to_16k(&samples, spec.sample_rate);
    
    let chunk_size = 16000;
    let hop_size = 8000;
    let mut output_audio = Vec::new();

    let mut start = 0;
    // Process only first 2 seconds (32000 samples) for speed
    let limit_samples = 32000.min(samples_16k.len());
    
    while start + chunk_size < limit_samples {
        let end = start + chunk_size;
        let chunk = &samples_16k[start..end];
        
        println!("Processing chunk {}-{}...", start, end);
        let out_chunk = pipeline.process(chunk, hop_size, 0.0, 0.3)?; // Pitch shift 0, Index rate 0.3
        
        // Log stats
        let max_val = out_chunk.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("  Output len: {}, Max Amp: {:.4}", out_chunk.len(), max_val);
        
        output_audio.extend_from_slice(&out_chunk);
        start += hop_size;
    }

    // Save output
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000, // We resampled output to 16k in pipeline
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for &sample in &output_audio {
        let amp = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(amp)?;
    }
    writer.finalize()?;
    println!("Saved to {}", output_path);

    Ok(())
}
