import soundfile as sf
import numpy as np
from argparse import ArgumentParser
import os
import glob
from scipy import signal
import pathlib
from util.utils import get_grain_labels, plot_umap, stitch_images_dir
from util.audio_utils import load_audio_and_resample
from util.audio_feature_helper import compute_audio_features, FEATURE_NAME_MAP
import umap
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
font = {'size': 16}
matplotlib.rc('font', **font)
from PIL import Image

def cluster_UMAP(audio_path, output_path=None, show=True, feature_list=['zcr', 'centroid', 'bandwidth', 'rolloff', 'mfcc']):
    """
    Extract features from an audio file and perform UMAP clustering.
    """
    target_sr = 22050  # Set your desired sample rate
    audio, sr = load_audio_and_resample(audio_path, target_sr)
    tar_l = 2048        # Set your desired grain length
    hop_size = 512
    n_frames = (len(audio) - tar_l) // hop_size + 1
    if n_frames < 1:
        n_frames = 1
    pad_len = (n_frames - 1) * hop_size + tar_l - len(audio)
    if pad_len > 0:
        audio = np.pad(audio, (0, pad_len))

    ola_window = signal.windows.hann(tar_l, sym=False)
    ola_windows = np.tile(ola_window, (n_frames, 1))
    ola_windows[0, :tar_l // 2] = ola_window[tar_l // 2]
    ola_windows[-1, tar_l // 2:] = ola_window[tar_l // 2]
    all_grains = []

    # Load label data
    json_path = audio_path.replace('.wav', '_enhanced.json')
    with open(json_path, 'r') as f:
        segments = json.load(f)

    for i in range(n_frames):
        window = ola_windows[i]
        start = i * hop_size
        end = start + tar_l
        chunk = audio[start:end] * window
        grain_feature = compute_audio_features(chunk, sr, feature_list)
        all_grains.append(grain_feature)
    all_grains = np.array(all_grains)
    print(f"Extracted {len(all_grains)} grains from audio.")

    # Perform UMAP
    reducer = umap.UMAP(n_components=2)
    all_grains_2d = reducer.fit_transform(all_grains)

    # Extract labels
    grain_labels = get_grain_labels(segments, n_frames, hop_size, tar_l, sr)

    plot_umap(all_grains_2d, grain_labels, output_path, feature_list, show=show)
    return all_grains_2d

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--export_dir', default="outputs/generations_luis", type=str)
    parser.add_argument('--audio', default="../stopping_enhanced/S-Br-T-th1-1.wav")
    parser.add_argument('--do_single', default=1, type=int)
    args = parser.parse_args()

    if os.path.exists(args.export_dir) is False:
        os.makedirs(args.export_dir)

    path = pathlib.Path(args.audio)
    out_path = os.path.join("umap_exp")
    if args.do_single > 0:
        for feature in FEATURE_NAME_MAP.keys():
            cluster_UMAP(args.audio, out_path, show=False, feature_list=[feature])
        stitch_images_dir()
    else:
        cluster_UMAP(args.audio, out_path, show=False)
