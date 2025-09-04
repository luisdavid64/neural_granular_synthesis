
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

import torch
from torch import nn

from models import hierarchical_model

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def encode_and_resynthesize(model, audio_path, output_path=None, use_cond=False):
    # Load audio
    audio, sr = sf.read(audio_path)
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # Audio is 44.8 kHz
    # Resample if needed
    target_sr = model.w_model.hparams.sr
    if sr != target_sr:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * target_sr / sr))
        sr = target_sr
    # Crop or pad to model's expected length
    model.adjust_target_length(len(audio))
    model.to(device)
    # Convert to tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    audio_tensor = audio_tensor.to(device)
    # Encode and decode
    use_cond=False
    with torch.no_grad():
        audio_recon = None
        if use_cond:
        # encoder_outputs = model.w_model.encode(audio_tensor)
            conds = torch.zeros(1).long().to(device) + 4 
            # conds = torch.zeros(1).long().to(device) + 0
            audio_recon, _, _, _ = model(audio_tensor, conds)
        else:
            encoder_outputs = model.w_model.encode(audio_tensor)
            audio_recon = model.w_model.decode(encoder_outputs["z"])
        audio_recon = audio_recon.cpu().squeeze().numpy()

            
    # Save or return
    if output_path:
        sf.write(output_path, audio_recon, sr)
        print("Resynthesized audio saved to:", output_path)
    return audio_recon


if __name__ == "__main__":
    
    # check issues in mostly silent interpolation ?
    # assert if fine-tuning (stage 3) degrades the model generative performances
    
    parser = ArgumentParser()
    parser.add_argument('--latent_mname', default="embedding_cond_l_E256_1LSTM", type=str)
    parser.add_argument('--waveform_mname', default="fulldrums_22k_L2048_Rej0202_normola", type=str)
    # parser.add_argument('--finetuned', action='store_true')
    parser.add_argument('--finetuned', default=0, type=int)
    parser.add_argument('--model_dir', default="outputs", type=str)
    parser.add_argument('--export_dir', default="outputs/generations", type=str)
    parser.add_argument('--samples_id', default=0, type=int)
    parser.add_argument('--temperature', default=1., type=float)
    parser.add_argument('--interp_len', default=4., type=float)
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
    encode_and_resynthesize(model, "/Users/luisreyes/Desktop/samples/audio1.wav", "output_resynth.wav")
