use std::path::Path;
use std::fs::File;
use hound;
use candle_core::Device;
use crate::realtime_pipeline::{RvcPipeline, InferenceBackend};

mod audio_processing;
mod feature_extractor;
mod pitch_extractor;
mod audio_stitching;
mod realtime_pipeline;
mod ring_buffer;
mod rvc_engine;

fn main() -> anyhow::Result<()> {
    println!("=== RVC Inference Backend Benchmark ===\n");

    let content_vec_path = "pretrain/content_vec_500.onnx";
    let rmvpe_path = "pretrain/rmvpe.onnx";
    let rvc_path = "models/Yukina_v2_merged.onnx";
    let index_path: Option<&str> = None;
    let device = Device::Cpu;

    // Load test audio
    let input_path = "assets/test.wav";
    println!("Loading test audio: {}", input_path);
    let mut reader = hound::WavReader::open(input_path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let samples_16k = audio_processing::resample_to_16k(&samples, spec.sample_rate);
    let chunk = &samples_16k[0..16000.min(samples_16k.len())];

    println!("Audio duration: {:.2}s", chunk.len() as f32 / 16000.0);
    println!();

    // Benchmark configurations
    let backends = vec![
        ("Candle CPU", InferenceBackend::Candle, true),
        #[cfg(feature = "onnxruntime")]
        ("ONNX Runtime", InferenceBackend::OnnxRuntime, true),
    ];

    let mut results = Vec::new();

    for (name, backend, enabled) in backends {
        if !enabled {
            continue;
        }

        println!("--- Testing: {} ---", name);

        let mut pipeline = match RvcPipeline::new(
            content_vec_path,
            rmvpe_path,
            rvc_path,
            index_path,
            device.clone(),
            backend,
        ) {
            Ok(p) => p,
            Err(e) => {
                println!("  ✗ Failed to load: {}", e);
                println!();
                continue;
            }
        };

        // Warmup
        print!("  Warmup... ");
        let _ = pipeline.process(chunk, 8000, 0.0, 0.0);
        println!("done");

        // Benchmark (3 runs)
        let mut times = Vec::new();
        for i in 0..3 {
            print!("  Run {}... ", i + 1);
            let start = std::time::Instant::now();
            let output = pipeline.process(chunk, 8000, 0.0, 0.0)?;
            let elapsed = start.elapsed();
            times.push(elapsed.as_secs_f32());
            println!("{:.3}s (output: {} samples)", elapsed.as_secs_f32(), output.len());
        }

        let avg_time = times.iter().sum::<f32>() / times.len() as f32;
        let realtime_factor = avg_time / (chunk.len() as f32 / 16000.0);

        println!("  Average: {:.3}s", avg_time);
        println!("  Real-time factor: {:.2}x", realtime_factor);
        println!("  Throughput: {:.2}x realtime", 1.0 / realtime_factor);
        println!();

        results.push((name, avg_time, realtime_factor));
    }

    // Summary
    println!("=== Summary ===");
    println!("{:<20} {:>12} {:>12} {:>15}", "Backend", "Time (s)", "RT Factor", "Throughput");
    println!("{:-<20} {:->12} {:->12} {:->15}", "", "", "", "");

    for (name, time, rt_factor) in &results {
        let throughput = 1.0 / rt_factor;
        let status = if *rt_factor < 1.0 { "✓" } else { "✗" };
        println!("{:<20} {:>11.3}s {:>11.2}x {:>14.2}x {}",
                 name, time, rt_factor, throughput, status);
    }

    // Performance comparison
    if results.len() >= 2 {
        let speedup = results[0].1 / results[1].1;
        println!();
        println!("Speed improvement: {:.1}x faster with {}",
                 speedup, results[1].0);
    }

    Ok(())
}
