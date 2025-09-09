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
}

model_path = dac.utils.download(model_type="16khz")
model = dac.DAC.load(model_path)
model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False
    
def compute_audio_features(chunk, sr, feature_list, normalize=True):
    """
    Compute selected audio features from librosa for a given chunk.
    feature_list: list of short feature names (e.g. ['zcr', 'centroid', ...])
    Returns a dict mapping feature names to numpy arrays.
    """
    features = {}
    # Nyquist frequency
    FS = sr / 2
    for short_name in feature_list:
        name = FEATURE_NAME_MAP.get(short_name, short_name)
        if name == 'zero_crossing_rate':
            val = librosa.feature.zero_crossing_rate(y=chunk).mean()
            features[name] = np.array([val])
        elif name == 'spectral_centroid':
            val = librosa.feature.spectral_centroid(y=chunk, sr=sr).mean()
            if normalize:
                val /= FS
            features[name] = np.array([val])
        elif name == 'spectral_bandwidth':
            val = librosa.feature.spectral_bandwidth(y=chunk, sr=sr).mean()
            if normalize:
                val /= FS
            features[name] = np.array([val])
        elif name == 'spectral_rolloff':
            val = librosa.feature.spectral_rolloff(y=chunk, sr=sr).mean()
            if normalize:
                val /= FS
            features[name] = np.array([val])
        elif name == 'mfcc':
            mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            features[name] = np.mean(mfccs, axis=1)
        elif name == 'spectral_flatness':
            val = librosa.feature.spectral_flatness(y=chunk).mean()
            features[name] = np.array([val])
        elif name == 'rms':
            val = librosa.feature.rms(y=chunk).mean()
            features[name] = np.array([val])
        elif name == 'spectral_flux':
            S = np.abs(librosa.stft(chunk))
            flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)).mean()
            features[name] = np.array([flux])
        elif name == 'spectral_sharpness':
            val = timbral_sharpness_numpy(chunk, sr)
            features[name] = np.array([val])
        elif name == 'spectral_roughness':
            val = timbral_roughness_numpy(chunk, sr)
            features[name] = np.array([val])
        elif name == 'clap':
            from clap_utils import ClapEmbedder
            embedder = ClapEmbedder()
            emb = embedder.embed_numpy(chunk, sr=sr, batch_size=1, max_length_s=10.0, pool_long="mean")
            features[name] = emb.flatten()
        elif name == 'dac':
            # Resample with librosa to 16kHz if needed
            if sr != 16000:
                chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
                sr = 16000
            chunk = audiotools.AudioSignal(chunk, sr)
            chunk.to(device)
            x = model.preprocess(chunk.audio_data, chunk.sample_rate)
            z, codes, latents, _, _ = model.encode(x)
            z = z.cpu().numpy()
            features[name] = z.flatten()
        else:
            raise ValueError(f"Unknown feature name: {short_name}")
    return features

def compute_metric(chunk_1, chunk_2, sr, features):
    """
    Compute a set of audio metrics from the extracted features.
    """
    metrics = {}
    features_1 = compute_audio_features(chunk_1, sr, features)
    features_2 = compute_audio_features(chunk_2, sr, features)

    # Compute additional metrics comparing features_1 and features_2
    for name, values in features_2.items():
        if name in ["clap", "dac"]:
            # Cosine similarity for high-dimensional features
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(features_1[name].reshape(1, -1), features_2[name].reshape(1, -1))[0][0]
            metrics[f"{name}"] = sim
        else:
            # Absolute difference for scalar features
            diff = np.abs(features_1[name] - features_2[name])
            metrics[f"{name}"] = diff.mean()
    return metrics

if __name__ == "__main__":
    # Example usage
    import soundfile as sf
    audio_path_1 = "../audio/drums_snote_crash.wav"
    audio_path_2 = "../audio/drums_snote_kick.wav"
    y1, sr1 = sf.read(audio_path_1)
    y2, sr2 = sf.read(audio_path_2)
    assert sr1 == sr2, "Sample rates must match"
    features = ['zcr', 'centroid', 'bandwidth', 'rolloff', 'mfcc', 'sharpness', 'roughness', 'clap', 'dac']
    metrics = compute_metric(y1, y2, sr1, features)
    print(metrics)