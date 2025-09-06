import librosa
import soundfile as sf
import torch
import numpy as np
import scipy
from argparse import ArgumentParser
import os
import glob
import numpy as np
import soundfile as sf
from scipy import signal
import pathlib
import torch
from torch import nn
from util.audio_utils import load_audio_and_resample
from util.audio_feature_helper import compute_audio_features, FEATURE_NAME_MAP
from util.utils import get_grain_labels, plot_umap, stitch_images_dir
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

def cluster_many(model, audio_path, audio_path_music, audio_path_ambient, output_path=None, show=True, do_encoding=True, feature_list=['zcr', 'centroid', 'bandwidth', 'rolloff', 'mfcc']):
    """
    Resynthesize an entire audio file using the pretrained model, with overlap-add batching.
    """
    print("Using features:", feature_list)
    output_path = "umap_exp_two_audios"
    model.to(device)
    audio, sr = sf.read(audio_path)
    tar_l = model.w_model.tar_l
    target_sr = model.w_model.hparams.sr
    audio, sr = load_audio_and_resample(audio_path, target_sr)
    music_audio, music_sr = load_audio_and_resample(audio_path_music, target_sr)
    ambient_audio, ambient_sr = load_audio_and_resample(audio_path_ambient, target_sr)
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
    n_frames_music = (len(music_audio) - tar_l) // hop_size + 1
    if n_frames_music < 1:
        n_frames_music = 1
    pad_len_music = (n_frames_music - 1) * hop_size + tar_l - len(music_audio)
    if pad_len_music > 0:
        music_audio = np.pad(music_audio, (0, pad_len_music))  
    n_frames_ambient = (len(ambient_audio) - tar_l) // hop_size + 1
    if n_frames_ambient < 1:
        n_frames_ambient = 1
    pad_len_ambient = (n_frames_ambient - 1) * hop_size + tar_l - len(ambient_audio)
    if pad_len_ambient > 0:
        ambient_audio = np.pad(ambient_audio, (0, pad_len_ambient))

    all_grains = []
    grain_labels = []
    print("Doing encoding:", do_encoding)
    with torch.no_grad():
        for i in range(n_frames):
            start = i * hop_size
            end = start + tar_l
            chunk = audio[start:end]
            chunk = librosa.util.fix_length(chunk, size=tar_l)
            if np.mean(np.abs(chunk)) < 0.01:
                continue
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            if do_encoding:
                encoder_outputs = model.w_model.encode(audio_tensor)
                all_grains += list(encoder_outputs["mu"].cpu().squeeze().numpy())
            else:
                # Use the feature handler for extensible feature extraction
                grain_feature = compute_audio_features(chunk, sr, feature_list)
                all_grains.append(grain_feature)
                grain_labels.append("tissue sonification")
    with torch.no_grad():
        for i in range(n_frames_music):
            start = i * hop_size
            end = start + tar_l
            chunk = music_audio[start:end]
            chunk = librosa.util.fix_length(chunk, size=tar_l)
            if np.mean(np.abs(chunk)) < 0.01:
                continue
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            if do_encoding:
                encoder_outputs = model.w_model.encode(audio_tensor)
                all_grains += list(encoder_outputs["mu"].cpu().squeeze().numpy())
            else:
                # Use the feature handler for extensible feature extraction
                grain_feature = compute_audio_features(chunk, sr, feature_list)
                all_grains.append(grain_feature)
                grain_labels.append("music")
    with torch.no_grad():
        for i in range(n_frames_ambient):
            start = i * hop_size
            end = start + tar_l
            chunk = ambient_audio[start:end]
            chunk = librosa.util.fix_length(chunk, size=tar_l)
            if np.mean(np.abs(chunk)) < 0.01:
                continue
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            if do_encoding:
                encoder_outputs = model.w_model.encode(audio_tensor)
                all_grains += list(encoder_outputs["mu"].cpu().squeeze().numpy())
            else:
                # Use the feature handler for extensible feature extraction
                grain_feature = compute_audio_features(chunk, sr, feature_list)
                all_grains.append(grain_feature)
                grain_labels.append("ambient")

    print(f"Extracted {len(all_grains)} grains from audio.")
    print(f"Labels: {set(grain_labels)}")
    """ Perform UMAP on grains """
    reducer = umap.UMAP(n_components=2)
    all_grains_2d = reducer.fit_transform(all_grains)
    # Plot UMAP with colors
    plot_umap(all_grains_2d, grain_labels, output_path, feature_list, show=show)
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
    parser.add_argument('--audio_music', default="/Users/luisreyes/Sonify/neural_granular_synthesis/audio/extrapol_clarinet_violin.wav")
    parser.add_argument('--audio_ambient', default="/Users/luisreyes/Sonify/neural_granular_synthesis/audio/rain.wav")
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
    out_path = os.path.join("umap_exp_two_audios")
    if args.do_single > 0:
        # Go through features one by one
        for feature in FEATURE_NAME_MAP.keys():
            cluster_many(model, args.audio, args.audio_music, args.audio_ambient, out_path, do_encoding=args.do_encoding > 0, show=False, feature_list=[feature])
        stitch_images_dir(img_dir=out_path, output_filename=os.path.join('umap_feature_grid_two_audios.png'))
    else:
        cluster_many(model, args.audio, args.audio_music, args.audio_ambient, output_path=out_path, do_encoding=args.do_encoding > 0, show=False)
