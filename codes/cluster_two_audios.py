import os
from argparse import ArgumentParser
import numpy as np
import librosa
import umap
import matplotlib
from util.audio_utils import load_audio_and_resample
from util.audio_feature_helper import compute_audio_features, FEATURE_NAME_MAP
from util.utils import plot_umap, stitch_images_dir

matplotlib.rc('font', size=16)

def cluster_many(audio_path, audio_path_music, audio_path_ambient, output_path=None, show=True, feature_list=['zcr', 'centroid', 'bandwidth', 'rolloff', 'mfcc']):
    print("Using features:", feature_list)
    output_path = output_path or "umap_exp_two_audios"
    target_sr = 22050
    tar_l = 2048
    hop_size = 512

    audio, sr = load_audio_and_resample(audio_path, target_sr)
    music_audio, _ = load_audio_and_resample(audio_path_music, target_sr)
    ambient_audio, _ = load_audio_and_resample(audio_path_ambient, target_sr)

    def pad_audio(aud, tar_l, hop_size):
        n_frames = max(1, (len(aud) - tar_l) // hop_size + 1)
        pad_len = (n_frames - 1) * hop_size + tar_l - len(aud)
        if pad_len > 0:
            aud = np.pad(aud, (0, pad_len))
        return aud, n_frames

    audio, n_frames = pad_audio(audio, tar_l, hop_size)
    music_audio, n_frames_music = pad_audio(music_audio, tar_l, hop_size)
    ambient_audio, n_frames_ambient = pad_audio(ambient_audio, tar_l, hop_size)

    all_grains = []
    grain_labels = []

    def extract_grains(aud, n_frames, label):
        for i in range(n_frames):
            start = i * hop_size
            end = start + tar_l
            chunk = librosa.util.fix_length(aud[start:end], size=tar_l)
            if np.mean(np.abs(chunk)) < 0.01:
                continue
            grain_feature = compute_audio_features(chunk, target_sr, feature_list)
            all_grains.append(grain_feature)
            grain_labels.append(label)

    extract_grains(audio, n_frames, "tissue sonification")
    extract_grains(music_audio, n_frames_music, "music")
    extract_grains(ambient_audio, n_frames_ambient, "ambient")

    print(f"Extracted {len(all_grains)} grains from audio.")
    print(f"Labels: {set(grain_labels)}")

    reducer = umap.UMAP(n_components=2)
    all_grains_2d = reducer.fit_transform(all_grains)
    plot_umap(all_grains_2d, grain_labels, output_path, feature_list, show=show)
    return all_grains_2d

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--audio', default="../stopping_enhanced/S-Br-T-th1-1.wav")
    parser.add_argument('--audio_music', default="/Users/luisreyes/Sonify/neural_granular_synthesis/audio/extrapol_clarinet_violin.wav")
    parser.add_argument('--audio_ambient', default="/Users/luisreyes/Sonify/neural_granular_synthesis/audio/rain.wav")
    parser.add_argument('--do_single', default=1, type=int)
    args = parser.parse_args()
    out_path = "umap_exp_two_audios"
    if args.do_single > 0:
        for feature in FEATURE_NAME_MAP.keys():
            cluster_many(args.audio, args.audio_music, args.audio_ambient, out_path, show=False, feature_list=[feature])
        stitch_images_dir(img_dir=out_path, output_filename='umap_feature_grid_two_audios.png')
    else:
        cluster_many(args.audio, args.audio_music, args.audio_ambient, output_path=out_path, show=False)
