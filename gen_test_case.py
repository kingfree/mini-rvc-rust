import numpy as np
import librosa
import torch

def generate_test_case():
    # 1. Sine wave
    sr = 16000
    duration = 0.1 # 100ms
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # 2. Librosa Mel Spectrogram (RVC settings)
    # n_fft=1024, hop=160, win=1024, window='hann', center=True
    mel = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=1024, 
        hop_length=160, 
        win_length=1024, 
        n_mels=128, 
        fmin=0, 
        fmax=8000,
        center=True
    )
    
    # RVC specific scaling
    mel_db = librosa.power_to_db(mel)
    mel_scaled = (mel_db + 20) / 20
    
    # Save files
    audio.tofile("assets/test_audio_sine.bin")
    mel_scaled.astype(np.float32).tofile("assets/test_mel_golden.bin")
    
    print(f"Audio shape: {audio.shape}")
    print(f"Mel shape: {mel_scaled.shape}")

if __name__ == "__main__":
    generate_test_case()
