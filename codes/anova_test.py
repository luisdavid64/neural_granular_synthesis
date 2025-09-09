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
from scipy.stats import f_oneway
font = {'size': 16}
matplotlib.rc('font', **font)
from PIL import Image

label_mapping = {
    "silence": 0,
    "skin": 1,
    "muscle": 2,
    "fat": 3,
    "brain": 4,
    "tumor": 5,
}

def extract_grains_and_labels(audio_path, feature_list):
    target_sr = 22050
    audio, sr = load_audio_and_resample(audio_path, target_sr)
    tar_l = 2048
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
    feature_names_expanded = []

    json_path = audio_path.replace('.wav', '_enhanced.json')
    with open(json_path, 'r') as f:
        segments = json.load(f)

    for i in range(n_frames):
        window = ola_windows[i]
        start = i * hop_size
        end = start + tar_l
        chunk = audio[start:end] * window

        grain_feature = compute_audio_features(chunk, sr, feature_list)
        feature_vec = []

        for feat in feature_list:
            feat_key = FEATURE_NAME_MAP.get(feat, feat)
            val = grain_feature[feat_key]
            if isinstance(val, (np.ndarray, list)):
                val = np.asarray(val).flatten()
                for j, v in enumerate(val):
                    feature_vec.append(v)
                    if i == 0:  # add names only once
                        feature_names_expanded.append(f"{feat_key}_{j}")
            else:
                feature_vec.append(val)
                if i == 0:
                    feature_names_expanded.append(feat_key)

        all_grains.append(feature_vec)

    all_grains = np.array(all_grains)
    grain_labels = get_grain_labels(segments, n_frames, hop_size, tar_l, sr)
    return all_grains, grain_labels, feature_names_expanded

def perform_anova(features, labels, feature_names):
    results = {}
    labels = np.array(labels)
    for i, fname in enumerate(feature_names):
        groups = [features[labels == label, i] for label in np.unique(labels)]
        f_val, p_val = f_oneway(*groups)
        
        # Effect size (eta-squared)
        ss_between = sum(len(g) * (np.mean(g) - np.mean(features[:, i]))**2 for g in groups)
        ss_total = sum(((g - np.mean(features[:, i]))**2).sum() for g in groups)
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        results[fname] = {
            'f_value': float(f_val),
            'p_value': float(p_val),
            'eta_sq': float(eta_sq)
        }
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--export_dir', default="outputs/generations_luis", type=str)
    parser.add_argument('--audio_folder', default="../stopping_enhanced/", type=str)
    args = parser.parse_args()

    if os.path.exists(args.export_dir) is False:
        os.makedirs(args.export_dir)

    feature_list = list(FEATURE_NAME_MAP.keys())
    print("Using features:", feature_list)
    all_features = []
    all_labels = []

    wav_files = glob.glob(os.path.join(args.audio_folder, "*.wav"))
    feature_list_expanded = None
    for wav_file in wav_files:
        print(f"Processing {wav_file}...")
        grains, labels, feature_list_expanded = extract_grains_and_labels(wav_file, feature_list)
        all_features.append(grains)
        all_labels.extend(labels)
        print(f"Extracted {len(grains)} grains.")
    feature_list = feature_list_expanded
    print("Expanded feature list:", feature_list)
    all_features = np.vstack(all_features)
    # Add dummy feature at the end
    dummy_feature = np.zeros((all_features.shape[0], 1))
    all_features = np.hstack((all_features, dummy_feature))
    all_labels = np.array(all_labels)
    feature_list.append("dummy_feature")

    anova_results = perform_anova(all_features, all_labels, feature_list)
    print("ANOVA results per feature:")
    for fname, res in anova_results.items():
        print(f"{fname}: F={res['f_value']:.4f}, p={res['p_value']:.4e}, eta²={res['eta_sq']:.4f}")

    # Optionally, save results to a file
    with open(os.path.join(args.export_dir, "anova_results.json"), "w") as f:
        json.dump(anova_results, f, indent=2)
