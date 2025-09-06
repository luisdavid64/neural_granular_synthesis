import soundfile as sf
import numpy as np
from scipy.signal import resample

def load_audio_and_resample(audio_path, target_sr):
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resample(audio, int(len(audio) * target_sr / sr))
        sr = target_sr
    return audio, sr