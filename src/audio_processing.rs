use ndarray::{Array1, Array2, Axis};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::fs::File;
use std::io::Read;

pub struct MelSpectrogram {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    window: Array1<f32>,
    mel_filterbank: Array2<f32>,
}

impl MelSpectrogram {
    pub fn new(
        _sample_rate: u32,
        n_fft: usize,
        hop_length: usize,
        win_length: usize,
        n_mels: usize,
    ) -> Self {
        let window = Array1::from_iter((0..win_length).map(|i| {
            0.5 * (1.0 - (2.0 * PI * i as f32 / (win_length - 1) as f32).cos())
        }));

        // Load pre-generated mel filterbank
        let mut mel_filterbank = Array2::zeros((n_mels, n_fft / 2 + 1));
        if let Ok(mut file) = File::open("assets/mel_filterbank.bin") {
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).unwrap();
            let data: &[f32] = unsafe {
                std::slice::from_raw_parts(buffer.as_ptr() as *const f32, buffer.len() / 4)
            };
            for i in 0..n_mels {
                for j in 0..(n_fft / 2 + 1) {
                    mel_filterbank[[i, j]] = data[i * (n_fft / 2 + 1) + j];
                }
            }
        }
        
        Self {
            n_fft,
            hop_length,
            win_length,
            n_mels,
            window,
            mel_filterbank,
        }
    }

    pub fn forward(&self, audio: &[f32]) -> anyhow::Result<Array2<f32>> {
        // Ensure audio length is sufficient
        if audio.len() < self.win_length {
            return Ok(Array2::zeros((self.n_mels, 1)));
        }
        
        let n_frames = (audio.len() - self.win_length) / self.hop_length + 1;
        // RMVPE UNet downsamples 4 times (2^4 = 16), so frames should be multiple of 16
        let n_frames_padded = (n_frames + 15) / 16 * 16;
        
        let mut power_spec = Array2::zeros((n_frames_padded, self.n_fft / 2 + 1));

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.n_fft);

        for i in 0..n_frames {
            let start = i * self.hop_length;
            let end = (start + self.win_length).min(audio.len());
            let frame = &audio[start..end];

            let mut buffer: Vec<Complex<f32>> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(&x, &w)| Complex::new(x * w, 0.0))
                .collect();
            
            if buffer.len() < self.n_fft {
                buffer.resize(self.n_fft, Complex::new(0.0, 0.0));
            }

            fft.process(&mut buffer);

            for j in 0..(self.n_fft / 2 + 1) {
                power_spec[[i, j]] = buffer[j].norm_sqr();
            }
        }
        
        // Padded frames remain zero

        // Apply mel filterbank: [n_mels, n_fft/2+1] * [n_frames, n_fft/2+1].T -> [n_mels, n_frames]
        let mel_spec = self.mel_filterbank.dot(&power_spec.reversed_axes());
        
        // Log-mel
        let log_mel_spec = mel_spec.mapv(|x| (x.max(1e-10)).log10());
        
        Ok(log_mel_spec)
    }
}

pub fn resample_to_16k(audio: &[f32], from_sr: u32) -> Vec<f32> {
    if from_sr == 16000 {
        return audio.to_vec();
    }
    if from_sr == 48000 {
        return audio.iter().step_by(3).cloned().collect();
    }
    audio.to_vec()
}
