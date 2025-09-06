import soundfile as sf
import torch
import numpy as np
from argparse import ArgumentParser
import os
import glob
import numpy as np
import soundfile as sf
from scipy import signal
import pathlib
import torch
from torch import nn
from util.utils import stitch_images_dir
from util.audio_utils import load_audio_and_resample
from util.audio_feature_helper import compute_audio_features, FEATURE_NAME_MAP
from models import hierarchical_model
import umap
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
font = {
        'size'   : 16
}
matplotlib.rc('font', **font)
from PIL import Image


def cluster_UMAP(model, audio_path, output_path=None, show=True, do_encoding=True, feature_list=['zcr', 'centroid', 'bandwidth', 'rolloff', 'mfcc']):
    """
    Resynthesize an entire audio file using the pretrained model, with overlap-add batching.
    """
    model.to(device)
    target_sr = model.w_model.hparams.sr
    audio, sr = load_audio_and_resample(audio_path, target_sr)
    tar_l = model.w_model.tar_l
    # hop_size = tar_l // 2  # 50% overlap
    hop_size = 512
    n_frames = (len(audio) - tar_l) // hop_size + 1
    # audio = signal.medfilt(audio, )  # simple denoising
    if n_frames < 1:
        n_frames = 1
    # Pad audio to fit full frames
    pad_len = (n_frames - 1) * hop_size + tar_l - len(audio)
    if pad_len > 0:
        audio = np.pad(audio, (0, pad_len))

    ola_window = signal.windows.hann(tar_l, sym=False)
    ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_frames, 1).type(torch.float32)
    ola_windows[0, :tar_l // 2] = ola_window[tar_l // 2]
    ola_windows[-1, tar_l // 2:] = ola_window[tar_l // 2]
    all_grains = []
    
    # Load label data
    json_path = audio_path.replace('.wav', '_enhanced.json')
    with open(json_path, 'r') as f:
        segments = json.load(f)
    print("Doing encoding:", do_encoding)
    with torch.no_grad():
        for i in range(n_frames):
            window = ola_windows[i].numpy()
            start = i * hop_size
            end = start + tar_l
            chunk = audio[start:end] * window
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            if do_encoding:
                encoder_outputs = model.w_model.encode(audio_tensor)
                all_grains += list(encoder_outputs["mu"].cpu().squeeze().numpy())
            else:
                # Use the feature handler for extensible feature extraction
                grain_feature = compute_audio_features(chunk, sr, feature_list)
                all_grains.append(grain_feature)
    all_grains = np.array(all_grains)
    print(f"Extracted {len(all_grains)} grains from audio.")

    """ Perform UMAP on grains """
    reducer = umap.UMAP(n_components=4)
    all_grains_2d = reducer.fit_transform(all_grains)
    
    """Extract labels"""

    grain_times = []
    for i in range(n_frames):
        start = i * hop_size
        end = start + tar_l
        # Use center time of grain in seconds
        center_sample = (start + end) // 2
        center_time = center_sample / sr
        grain_times.append(center_time)

    # Assign labels to grains
    grain_labels = []
    for t in grain_times:
        label = None
        for seg in segments:
            if seg['start_s'] <= t < seg['end_s']:
                label = seg['label']
                break
        grain_labels.append(label if label is not None else 'unknown')

    unique_labels = sorted(set(grain_labels))
    palette = sns.color_palette("tab10", len(unique_labels))
    label_to_color = {lab: palette[i] for i, lab in enumerate(unique_labels)}

    # Plot UMAP with colors
    plt.figure(figsize=(10, 10))
    for lab in unique_labels:
        idx = [i for i, l in enumerate(grain_labels) if l == lab]
        plt.scatter(all_grains_2d[idx, 0], all_grains_2d[idx, 1], s=20, alpha=0.6, label=lab, color=label_to_color[lab])
    # add features to title
    plt.title(f"UMAP of Audio Grains (Labeled) - Features: {', '.join(feature_list)}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.axis('equal')
    if show:
        plt.show()
    # Save fig name based on features used
    plt.savefig(os.path.join(output_path, f"umap_{'_'.join(feature_list)}.png"))
    plt.close()
    return all_grains_2d


if __name__ == "__main__":
    
    # check issues in mostly silent interpolation ?
    # assert if fine-tuning (stage 3) degrades the model generative performances
    
    parser = ArgumentParser()
    parser.add_argument('--latent_mname', default="embedding_cond_l_E256_1LSTM", type=str)
    parser.add_argument('--waveform_mname', default="fulldrums_22k_L2048_Rej0202_normola", type=str)
    parser.add_argument('--finetuned', default=0, type=int)
    parser.add_argument('--model_dir', default="outputs", type=str)
    parser.add_argument('--export_dir', default="outputs/generations_luis", type=str)
    parser.add_argument('--audio', default="../stopping_enhanced/S-Br-T-th1-1.wav")
    parser.add_argument('--audio_2', default=None, type=str)
    parser.add_argument('--do_encoding', default=0, type=int)
    parser.add_argument('--do_single', default=1, type=int)
    args = parser.parse_args()
    
    if os.path.exists(args.export_dir) is False:
        os.makedirs(args.export_dir)
    
    # loading pretrained models: either using finetuned checkpoint of hierarchical_model
    # or building hierarchical_model from pretrained w_model and l_model before finetuning
    
    args.latent_mname = args.waveform_mname+"__"+args.latent_mname
    
    w_ckpt_file = sorted(glob.glob(os.path.join(args.model_dir, args.waveform_mname, "checkpoints", "*.ckpt")))[-1]
    w_yaml_file = os.path.join(args.model_dir, args.waveform_mname, "hparams.yaml")
    l_ckpt_file = sorted(glob.glob(os.path.join(args.model_dir, args.latent_mname, "checkpoints", "*.ckpt")))[-1]
    l_yaml_file = os.path.join(args.model_dir, args.latent_mname, "hparams.yaml")
    
    if args.finetuned:
        args.mname = args.latent_mname+"__finetuned"
        print("\n*** loading finetuned model",args.mname)
        ckpt_file = sorted(glob.glob(os.path.join(args.model_dir,args.mname,"checkpoints","*.ckpt")))[-1]
        yaml_file = os.path.join(args.model_dir,args.mname,"hparams.yaml")
        model = hierarchical_model.load_from_checkpoint(checkpoint_path=ckpt_file,hparams_file=yaml_file,map_location='cpu',
                                                    w_ckpt_file=w_ckpt_file,w_yaml_file=w_yaml_file,l_ckpt_file=l_ckpt_file,l_yaml_file=l_yaml_file)
    else:
        args.mname = args.latent_mname
        print("\n*** loading pretrained waveform and latent models",args.waveform_mname,args.latent_mname)
        model = hierarchical_model(w_ckpt_file=w_ckpt_file,w_yaml_file=w_yaml_file,l_ckpt_file=l_ckpt_file,l_yaml_file=l_yaml_file)
    model.eval()
    path = pathlib.Path(args.audio)
    
    # Output path is composed of export dir + path.name + manual_ola.wav
    out_path = os.path.join("umap_exp")
    if args.do_single > 0:
        # Go through features one by one
        for feature in FEATURE_NAME_MAP.keys():
            cluster_UMAP(model, args.audio, out_path, do_encoding=args.do_encoding > 0, show=False, feature_list=[feature])
        stitch_images_dir()
    else:
        cluster_UMAP(model, args.audio, out_path, do_encoding=args.do_encoding > 0, show=False)
