from __future__ import division
import numpy as np
import soundfile as sf
import timbral_models.timbral_util as timbral_util
from timbral_models.Timbral_Sharpness import sharpness_Fastl

from scipy.signal import resample_poly

def _to_mono(x, fs, phase_correction=False):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        mono = x
    elif x.ndim == 2:
        # accept (n, ch) or (ch, n)
        if x.shape[0] < x.shape[1]:
            tf = x              # (n, ch)
        else:
            tf = x.T            # (ch, n) -> (n, ch)
        if phase_correction and hasattr(timbral_util, "phase_correct_sum"):
            mono = timbral_util.phase_correct_sum(tf, fs=fs)
        else:
            mono = np.mean(tf, axis=1)
    else:
        raise ValueError("samples must be 1D or 2D (n, ch)/(ch, n).")
    return mono

def _maybe_upsample(x, fs, target_fs=48000):
    if fs >= 44100:
        return x, fs
    # rational resample to target_fs
    g = np.gcd(int(target_fs), int(fs))
    up, down = target_fs // g, fs // g
    x_up = resample_poly(x, up, down)
    return x_up, target_fs

def timbral_sharpness_numpy(samples,
                            fs,
                            dev_output=False,
                            phase_correction=False,
                            clip_output=False):
    """
    NumPy-only timbral sharpness with automatic upsampling to avoid Bark-band index errors.
    """
    if fs is None or int(fs) <= 0:
        raise ValueError("fs must be a positive integer.")
    fs = int(fs)

    # mono
    mono = _to_mono(samples, fs, phase_correction=phase_correction)

    # early-out for silence
    if not np.any(mono):
        return [np.log10(1e-12)] if dev_output else 0.0

    # upsample preemptively if fs < 44.1k
    mono, fs = _maybe_upsample(mono, fs, target_fs=48000)

    # windowing (kept consistent with timbral_models)
    windowed_audio = timbral_util.window_audio(mono, window_length=4096)

    windowed_rms = []
    windowed_sharp = []
    for i in range(windowed_audio.shape[0]):
        w = windowed_audio[i, :]
        rms = float(np.sqrt(np.mean(w * w))) if w.size else 0.0
        windowed_rms.append(rms)

        if rms == 0.0:
            windowed_sharp.append(0.0)
            continue

        try:
            N_entire, N_single = timbral_util.specific_loudness(w, Pref=100.0, fs=fs, Mod=0)
        except IndexError:
            # Fallback: upsample to 48k and retry once (in case input was borderline)
            w_up, fs_up = _maybe_upsample(w, fs, target_fs=48000)
            N_entire, N_single = timbral_util.specific_loudness(w_up, Pref=100.0, fs=fs_up, Mod=0)

        sharp = sharpness_Fastl(N_single) if N_entire > 0 else 0.0
        windowed_sharp.append(sharp)

    windowed_rms = np.asarray(windowed_rms, dtype=float)
    windowed_sharp = np.asarray(windowed_sharp, dtype=float)

    if not np.any(windowed_rms):
        rms_sharp = 0.0
    else:
        rms_sharp = np.average(windowed_sharp, weights=windowed_rms * windowed_rms)

    # log mapping with guard
    eps = 1e-12
    rms_sharp_log = np.log10(max(rms_sharp, eps))

    if dev_output:
        return [rms_sharp_log]

    # preserve the original post-regression mapping
    all_metrics = np.ones(2)
    all_metrics[0] = rms_sharp_log
    coefficients = [102.50508921364404, 34.432655185001735]
    out = float(np.sum(all_metrics * coefficients))

    if clip_output:
        out = timbral_util.output_clip(out)

    return out