
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

from models import hierarchical_model

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def encode_and_resynthesize(model, audio_path, output_path=None, use_cond=False):
    """
    Resynthesize an entire audio file using the pretrained model, with overlap-add batching.
    """
    model.to(device)
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    target_sr = model.w_model.hparams.sr
    if sr != target_sr:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * target_sr / sr))
        sr = target_sr
    tar_l = model.w_model.tar_l
    hop_size = tar_l // 2  # 50% overlap
    n_frames = (len(audio) - tar_l) // hop_size + 1
    # audio = signal.medfilt(audio, )  # simple denoising
    if n_frames < 1:
        n_frames = 1
    # Pad audio to fit full frames
    pad_len = (n_frames - 1) * hop_size + tar_l - len(audio)
    if pad_len > 0:
        audio = np.pad(audio, (0, pad_len))
    output = np.zeros_like(audio, dtype=np.float32)
    norm = np.zeros_like(audio, dtype=np.float32)

    ola_window = signal.windows.hann(tar_l, sym=False)
    ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_frames, 1).type(torch.float32)
    ola_windows[0, :tar_l // 2] = ola_window[tar_l // 2]
    ola_windows[-1, tar_l // 2:] = ola_window[tar_l // 2]
    with torch.no_grad():
        for i in range(n_frames):
            window = ola_windows[i].numpy()
            start = i * hop_size
            end = start + tar_l
            chunk = audio[start:end] * window
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            if use_cond:
            # encoder_outputs = model.w_model.encode(audio_tensor)
                l_encoder_outputs, w_encoder_outputs = model.encode(audio_tensor, sampling=False)
                conds = torch.zeros(l_encoder_outputs["e"].shape[0]).long().to(device)
                audio_recon, _= model.decode(l_encoder_outputs["e"], conds=conds)
            else:
                encoder_outputs = model.w_model.encode(audio_tensor)
                print(encoder_outputs["z"].shape)
                audio_recon = model.w_model.decode(encoder_outputs["z"])
            # exit()
            audio_recon = audio_recon.cpu().squeeze().numpy()
            output[start:end] += audio_recon * window
            norm[start:end] += window ** 2
    # Avoid division by zero
    norm[norm == 0] = 1e-8
    output = output
    # Remove padding if any
    output = output[:len(audio) - pad_len] if pad_len > 0 else output
    if output_path:
        sf.write(output_path, output, sr)
    return output


if __name__ == "__main__":
    
    # check issues in mostly silent interpolation ?
    # assert if fine-tuning (stage 3) degrades the model generative performances
    
    parser = ArgumentParser()
    parser.add_argument('--latent_mname', default="embedding_cond_l_E256_1LSTM", type=str)
    parser.add_argument('--waveform_mname', default="fulldrums_22k_L2048_Rej0202_normola", type=str)
    # parser.add_argument('--finetuned', action='store_true')
    parser.add_argument('--finetuned', default=0, type=int)
    parser.add_argument('--model_dir', default="outputs", type=str)
    parser.add_argument('--export_dir', default="outputs/generations_luis", type=str)
    parser.add_argument('--samples_id', default=0, type=int)
    parser.add_argument('--temperature', default=1., type=float)
    parser.add_argument('--interp_len', default=4., type=float)
    parser.add_argument('--audio', default="/Users/luisreyes/Desktop/samples/audio1.wav")
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
    out_path = os.path.join(args.export_dir, path.stem + "_manual_ola.wav")
    encode_and_resynthesize(model, args.audio, out_path)
