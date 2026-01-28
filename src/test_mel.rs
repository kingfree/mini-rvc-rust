mod audio_processing;
mod feature_extractor;

use std::fs::File;
use std::io::Read;
use ndarray::Array2;

fn main() -> anyhow::Result<()> {
    println!("--- Mel Spectrogram Correctness Test ---");

    // 1. Load test audio
    let mut audio_file = File::open("assets/test_audio_sine.bin")?;
    let mut audio_buf = Vec::new();
    audio_file.read_to_end(&mut audio_buf)?;
    let audio: &[f32] = unsafe {
        std::slice::from_raw_parts(audio_buf.as_ptr() as *const f32, audio_buf.len() / 4)
    };

    // 2. Load golden mel
    let mut mel_file = File::open("assets/test_mel_golden.bin")?;
    let mut mel_buf = Vec::new();
    mel_file.read_to_end(&mut mel_buf)?;
    let golden_data: &[f32] = unsafe {
        std::slice::from_raw_parts(mel_buf.as_ptr() as *const f32, mel_buf.len() / 4)
    };
    let golden_mel = Array2::from_shape_vec((128, 11), golden_data.to_vec())?;

    // 3. Compute Rust mel
    let mel_extractor = audio_processing::MelSpectrogram::new(16000, 1024, 160, 1024, 128);
    let rust_mel = mel_extractor.forward(audio)?;

    println!("Rust Mel shape: {:?}", rust_mel.shape());
    println!("Golden Mel shape: {:?}", golden_mel.shape());

    // 4. Compare
    let mut diff = 0.0;
    let n_compare_frames = 11;
    for j in 0..n_compare_frames { // frames
        for i in 0..128 { // bins
            diff += (rust_mel[[i, j]] - golden_mel[[i, j]]).abs();
        }
    }
    let avg_diff = diff / (128.0 * n_compare_frames as f32);
    println!("Average absolute difference (11 frames): {}", avg_diff);

    if avg_diff < 0.1 {
        println!("✅ Mel Spectrogram Correctness Test Passed!");
    } else {
        println!("❌ Mel Spectrogram Correctness Test Failed!");
        println!("Golden Frame 0, Bin 0: {}", golden_mel[[0, 0]]);
        println!("Rust Frame 0, Bin 0: {}", rust_mel[[0, 0]]);
        println!("Golden Frame 1, Bin 0: {}", golden_mel[[0, 1]]);
        println!("Rust Frame 1, Bin 0: {}", rust_mel[[0, 1]]);
    }

    Ok(())
}
