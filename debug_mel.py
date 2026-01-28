import numpy as np
import librosa

def debug():
    # Sine wave
    sr = 16000
    duration = 0.1 # 100ms
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # 1. STFT
    stft = librosa.stft(audio, n_fft=1024, hop_length=160, win_length=1024, window='hann', center=True)
    spec = np.abs(stft)**2
    
    # 2. Mel
    mel_basis = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=128, fmin=0, fmax=8000)
    mel = np.dot(mel_basis, spec)
    
    print(f"STFT shape: {stft.shape}")
    print(f"Spec sample (0,0): {spec[0,0]}")
    print(f"Mel sample (0,0): {mel[0,0]}")
    print(f"Mel sample (0,1): {mel[0,1]}")
    
    # RVC specific scaling
    mel_db = librosa.power_to_db(mel)
    print(f"DB sample (0,0): {mel_db[0,0]}")
    print(f"DB sample (0,1): {mel_db[0,1]}")
    mel_scaled = (mel_db + 20) / 20
    print(f"Scaled Mel sample (0,0): {mel_scaled[0,0]}")

if __name__ == "__main__":
    debug()
