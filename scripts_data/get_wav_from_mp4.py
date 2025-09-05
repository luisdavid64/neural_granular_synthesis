"""
    Get wav files from mmp4 in folder
"""
import os
import glob
import subprocess
import soundfile as sf
import argparse
def get_wav_from_mp4(mp4_path, wav_path, sr=16000):
    command = f"ffmpeg -i {mp4_path} -ar {sr} -ac 1 -vn -y {wav_path}"
    subprocess.call(command, shell=True)
    print(f"Saved {wav_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="stopping_enhanced", help="Path to folder with mp4 files")
    parser.add_argument("--out_path", type=str, default="stopping_enhanced", help="Path to folder to save wav files")
    parser.add_argument("--sr", type=int, default=16000, help="Sampling rate for wav files")
    args = parser.parse_args()
    
    os.makedirs(args.out_path, exist_ok=True)
    mp4_files = glob.glob(os.path.join(args.data_path, "*.mp4"))
    print(f"Found {len(mp4_files)} mp4 files in {args.data_path}")
    for mp4_file in mp4_files:
        base_name = os.path.basename(mp4_file)
        wav_file = os.path.join(args.out_path, base_name[:-4] + ".wav")
        get_wav_from_mp4(mp4_file, wav_file, sr=args.sr)