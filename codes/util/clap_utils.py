import numpy as np
import torch
import librosa
from typing import List, Union
from transformers import ClapProcessor, ClapModel

TARGET_SR = 48_000  # CLAP expects 48 kHz
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _to_mono(y: np.ndarray) -> np.ndarray:
    """Accept (T,), (C,T), or (T, C); return mono (T,)."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        # handle either (C,T) or (T,C)
        if y.shape[0] < y.shape[1]:  # heuristic: (C,T)
            return np.mean(y, axis=0)
        else:  # (T,C)
            return np.mean(y, axis=1)
    raise ValueError("Audio must be 1D or 2D numpy array.")

def _resample_to_48k(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return y.astype(np.float32, copy=False)
    return librosa.resample(y.astype(np.float32, copy=False), orig_sr=sr, target_sr=TARGET_SR)

def _chunk_audio(y_48k: np.ndarray, max_length_s: float = 10.0) -> List[np.ndarray]:
    """
    CLAP (HTSAT) is trained with ~10s inputs. For longer clips, chunk and pool.
    Returns a list of non-overlapping chunks (last chunk padded if short).
    """
    max_len = int(TARGET_SR * max_length_s)
    if len(y_48k) <= max_len:
        return [y_48k]
    chunks = []
    for start in range(0, len(y_48k), max_len):
        chunk = y_48k[start:start+max_len]
        if len(chunk) == 0:
            break
        chunks.append(chunk)
    return chunks

class ClapEmbedder:
    def __init__(self, model_name: str = "laion/clap-htsat-fused"):
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(DEVICE).eval()

    @torch.no_grad()
    def embed_numpy(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        sr: int = 16_000,
        batch_size: int = 1,
        max_length_s: float = 10.0,
        pool_long: str = "mean",  # how to pool chunk embeddings: "mean" or "max"
    ) -> np.ndarray:
        """
        Parameters
        ----------
        audio : np.ndarray or list[np.ndarray]
            One array (T,) / (C,T) / (T,C), or a list of such arrays.
        sr : int
            Sampling rate of provided arrays (16 kHz in your case).
        batch_size : int
            Number of items (or chunks) per forward pass.
        max_length_s : float
            Chunk length for long audio (seconds).
        pool_long : str
            Pooling across chunks for long audio: "mean" or "max".
        Returns
        -------
        np.ndarray of shape (N, D)
            L2-normalized embeddings.
        """
        # Normalize input to a list
        items = audio if isinstance(audio, list) else [audio]

        # Preprocess: mono + resample + chunk
        batched_wave_lists = []  # each entry is a list of chunks for one original item
        for y in items:
            y = _to_mono(y)
            y48 = _resample_to_48k(y, sr)
            chunks = _chunk_audio(y48, max_length_s=max_length_s)
            batched_wave_lists.append(chunks)

        # Flatten chunks for batched inference
        flat_chunks = [chunk for chunks in batched_wave_lists for chunk in chunks]

        # Run model in batches over chunks
        all_chunk_embs: List[np.ndarray] = []
        for i in range(0, len(flat_chunks), batch_size):
            batch = flat_chunks[i:i+batch_size]
            inputs = self.processor(audios=batch, sampling_rate=TARGET_SR,
                                    return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            feats = self.model.get_audio_features(**inputs)  # (B, D)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            all_chunk_embs.append(feats.cpu().numpy())
        flat_embs = np.concatenate(all_chunk_embs, axis=0) if all_chunk_embs else np.zeros((0, 512), dtype=np.float32)

        # Unflatten: pool chunk embeddings back to per-item embeddings
        embs_per_item: List[np.ndarray] = []
        cursor = 0
        for chunks in batched_wave_lists:
            k = len(chunks)
            if k == 0:
                embs_per_item.append(np.zeros((1, flat_embs.shape[1]), dtype=np.float32))
                continue
            item_chunk_embs = flat_embs[cursor:cursor+k]
            cursor += k
            if pool_long == "mean":
                pooled = item_chunk_embs.mean(axis=0, keepdims=True)
            elif pool_long == "max":
                pooled = item_chunk_embs.max(axis=0, keepdims=True)
            else:
                raise ValueError("pool_long must be 'mean' or 'max'")
            # Ensure L2-norm = 1 after pooling
            norm = np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-12
            pooled = pooled / norm
            embs_per_item.append(pooled)

        embs = np.vstack(embs_per_item)
        return embs

