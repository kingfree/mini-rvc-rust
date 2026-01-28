import numpy as np
import librosa

def generate():
    mel = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=0, fmax=8000)
    # Save as binary f32
    mel.astype(np.float32).tofile("assets/mel_filterbank.bin")
    print("Mel filterbank generated: 128x513")

if __name__ == "__main__":
    generate()
