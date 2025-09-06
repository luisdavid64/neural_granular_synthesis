import numpy as np
import timbral_models.timbral_util as timbral_util
from scipy.signal import resample_poly

def _to_mono_numpy(x, fs, phase_correction=False):
    """
    Accepts (n,), (n, ch), or (ch, n) and returns mono (n,).
    Uses timbral_util.phase_correct_sum if available and requested.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        mono = x
    elif x.ndim == 2:
        # normalize to (n, ch)
        if x.shape[0] < x.shape[1]:
            tf = x            # (n, ch)
        else:
            tf = x.T          # (ch, n) -> (n, ch)
        if phase_correction and hasattr(timbral_util, "phase_correct_sum"):
            mono = timbral_util.phase_correct_sum(tf, fs=fs)
        else:
            mono = np.mean(tf, axis=1)
    else:
        raise ValueError("samples must be 1D mono or 2D (n, ch)/(ch, n).")
    return mono


def timbral_roughness_numpy(samples,
                            fs,
                            dev_output=False,
                            phase_correction=False,
                            clip_output=False,
                            peak_picking_threshold=0.01):
    """
    NumPy-only version of timbral_roughness (Vassilakis 2007), no file I/O.

    Parameters
    ----------
    samples : np.ndarray
        Audio array: (n,), (n, ch), or (ch, n). Float convertible.
    fs : int
        Sample rate (Hz).
    dev_output : bool
        If True, returns [mean_roughness] before regression; otherwise returns mapped roughness.
    phase_correction : bool
        If True and timbral_util.phase_correct_sum exists, use it for mono sum.
    clip_output : bool
        If True, clip the final mapped roughness with timbral_util.output_clip.
    peak_picking_threshold : float
        Threshold for timbral_util.detect_peaks (same as original).

    Returns
    -------
    float or list[float]
        If dev_output=False: mapped roughness.
        If dev_output=True: [mean_roughness] (pre-mapping).
    """
    if fs is None or int(fs) <= 0:
        raise ValueError("fs (sample rate) must be a positive integer.")
    fs = int(fs)

    # --- Read/prepare input (NumPy only) ---
    x = _to_mono_numpy(samples, fs, phase_correction=phase_correction)

    # Early-out for silence
    if not np.any(x):
        return [0.0] if dev_output else 0.0

    # --- Pad audio (same as original: 512 zeros at start) ---
    audio = np.pad(x, (512, 0), mode="constant")

    # --- Frame parameters (50 ms windows, 50% overlap) ---
    time_step = 0.05  # 50 ms
    step_samples = int(round(fs * time_step))
    step_samples = max(step_samples, 1)
    nfft = step_samples

    # Hamming window of length nfft (original uses np.hamming(nfft+2)[1:-1])
    if nfft >= 2:
        window = np.hamming(nfft + 2)[1:-1]
    else:
        window = np.ones(nfft, dtype=float)

    hop = nfft // 2  # 50% overlap
    if hop == 0:
        hop = 1

    audio_len = int(len(audio))

    # Number of frames (match original looping behavior)
    if audio_len > step_samples:
        num_frames = 1 + int(np.floor((audio_len - step_samples) / hop))
    else:
        num_frames = 1

    # Next power of two for zero-padding along FFT length
    if step_samples <= 1:
        next_pow_2 = 1
    else:
        next_pow_2 = 1 << int(np.ceil(np.log2(step_samples)))

    reshaped_audio = np.zeros((next_pow_2, num_frames), dtype=float)

    # --- Frame, window, and place into matrix ---
    for i in range(num_frames):
        start_idx = i * hop
        end_idx = start_idx + step_samples
        if end_idx <= audio_len:
            frame = audio[start_idx:end_idx]
        else:
            # last partial frame (if any)
            frame = audio[start_idx:audio_len]
            if frame.size == 0:
                break
            # zero-pad to step_samples for windowing
            frame = np.pad(frame, (0, step_samples - frame.size), mode="constant")

        # apply window length safely
        if frame.size == window.size:
            wframe = frame * window
        else:
            # in pathological cases, fall back to min length
            m = min(frame.size, window.size)
            wframe = np.zeros(step_samples)
            wframe[:m] = frame[:m] * window[:m]

        reshaped_audio[:step_samples, i] = wframe

    # --- Spectrum magnitude ---
    spec = np.fft.rfft(reshaped_audio, axis=0)  # rfft already gives up to Nyquist
    spec = np.abs(spec)
    spec_len = spec.shape[0]

    # Frequency axis for peak picking
    freq = np.linspace(0.0, fs / 2.0, spec_len)

    # --- Normalize spectrogram (guard for flat spectrum) ---
    smin = spec.min()
    smax = spec.max()
    if smax > smin:
        norm_spec = (spec - smin) / (smax - smin)
    else:
        norm_spec = np.zeros_like(spec)

    # --- Peak picking per frame (uses project’s util) ---
    cthr = float(peak_picking_threshold)
    _, no_segments = norm_spec.shape

    allpeaktime = []
    allpeaklevel = []

    for i in range(no_segments):
        d = norm_spec[:, i]
        d_un = spec[:, i]
        # timbral_util.detect_peaks returns (positions, levels, freqs)
        peak_pos, peak_level, peak_x = timbral_util.detect_peaks(d, cthr=cthr, unprocessed_array=d_un, freq=freq)
        allpeaktime.append(peak_x)
        allpeaklevel.append(peak_level)

    # --- Vassilakis roughness per frame ---
    def plomp(f1, f2):
        b1 = 3.51
        b2 = 5.75
        xstar = 0.24
        s1 = 0.0207
        s2 = 18.96
        s = np.tril(xstar / ((s1 * np.minimum(f1, f2)) + s2))
        pd = np.exp(-b1 * s * np.abs(f2 - f1)) - np.exp(-b2 * s * np.abs(f2 - f1))
        return pd

    allroughness = []
    for frame in range(len(allpeaklevel)):
        frame_freq = np.asarray(allpeaktime[frame], dtype=float)
        frame_level = np.asarray(allpeaklevel[frame], dtype=float)

        if frame_freq.size > 1:
            # Pairwise matrices
            f2 = np.kron(np.ones((frame_freq.size, 1)), frame_freq)
            f1 = f2.T
            v2 = np.kron(np.ones((frame_level.size, 1)), frame_level)
            v1 = v2.T

            X = v1 * v2
            Y = (2.0 * v2) / (v1 + v2 + 1e-12)   # guard divide-by-zero
            Z = plomp(f1, f2)
            rough = (X ** 0.1) * (0.5 * (Y ** 3.11)) * Z
            allroughness.append(float(np.sum(rough)))
        else:
            allroughness.append(0.0)

    mean_roughness = float(np.mean(allroughness)) if allroughness else 0.0

    if dev_output:
        return [mean_roughness]

    # --- Linear regression mapping (same as original) ---
    if mean_roughness < 0.01:
        out = 0.0
    else:
        out = np.log10(mean_roughness) * 13.98779569 + 48.97606571545886

    if clip_output and hasattr(timbral_util, "output_clip"):
        out = timbral_util.output_clip(out)

    return float(out)
