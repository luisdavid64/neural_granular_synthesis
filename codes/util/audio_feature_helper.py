import numpy as np
import librosa
from util.roughness import timbral_roughness_numpy
from util.sharpness import timbral_sharpness_numpy
import dac
import audiotools
import torch
# Let's add: sharpness and roughness
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

FEATURE_NAME_MAP = {
    'zcr': 'zero_crossing_rate',
    'centroid': 'spectral_centroid',
    'bandwidth': 'spectral_bandwidth',
    'rolloff': 'spectral_rolloff',
    'mfcc': 'mfcc',
    'flatness': 'spectral_flatness',
    'rms': 'rms',
    'flux': 'spectral_flux',
    'sharpness': 'spectral_sharpness',
    'roughness': 'spectral_roughness',
    "clap": "clap",
    "dac": "dac",
}

def compute_audio_features(chunk, sr, feature_list):
    """
    Compute selected audio features from librosa for a given chunk.
    feature_list: list of short feature names (e.g. ['zcr', 'centroid', ...])
    Returns a 1D numpy array of concatenated feature values.
    """
    features = []
    for short_name in feature_list:
        name = FEATURE_NAME_MAP.get(short_name, short_name)
        if name == 'zero_crossing_rate':
            val = librosa.feature.zero_crossing_rate(y=chunk).mean()
            features.append(val)
        elif name == 'spectral_centroid':
            val = librosa.feature.spectral_centroid(y=chunk, sr=sr).mean()
            features.append(val)
        elif name == 'spectral_bandwidth':
            val = librosa.feature.spectral_bandwidth(y=chunk, sr=sr).mean()
            features.append(val)
        elif name == 'spectral_rolloff':
            val = librosa.feature.spectral_rolloff(y=chunk, sr=sr).mean()
            features.append(val)
        elif name == 'mfcc':
            mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
        elif name == 'spectral_flatness':
            val = librosa.feature.spectral_flatness(y=chunk).mean()
            features.append(val)
        elif name == 'rms':
            val = librosa.feature.rms(y=chunk).mean()
            features.append(val)
        elif name == 'spectral_flux':
            S = np.abs(librosa.stft(chunk))
            flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)).mean()
            features.append(flux)
        elif name == 'spectral_sharpness':
            val = timbral_sharpness_numpy(chunk, sr)
            features.append(val)
        elif name == 'spectral_roughness':
            val = timbral_roughness_numpy(chunk, sr)
            features.append(val)
        elif name == 'clap':
            from util.clap_utils import ClapEmbedder
            embedder = ClapEmbedder()
            emb = embedder.embed_numpy(chunk, sr=sr, batch_size=1, max_length_s=10.0, pool_long="mean")
            features.extend(emb.flatten())
        elif name == 'dac':
            model_path = dac.utils.download(model_type="16khz")
            model = dac.DAC.load(model_path)
            model.to(device)
            # Resample with librosa to 16kHz if needed
            if sr != 16000:
                chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
                sr = 16000
            chunk = audiotools.AudioSignal(chunk, sr)
            chunk.to(device)
            x = model.preprocess(chunk.audio_data, chunk.sample_rate)
            z, codes, latents, _, _ = model.encode(x)
            z = z.cpu().numpy()
            features.extend(z.flatten())

        else:
            raise ValueError(f"Unknown feature name: {short_name}")
    return np.array(features)