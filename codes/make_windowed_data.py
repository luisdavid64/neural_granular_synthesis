import pandas as pd
from util.audio_feature_helper import FEATURE_NAME_MAP, compute_audio_features
import librosa
import numpy as np
import pandas as pd
from pathlib import Path


def make_windowed_data(audio_path, segments, feature_list,
                          win_len=2048, hop_len=512, sr=22050):
    """
    Convert audio + labeled segments into a windowed dataset of features.
    Args:
        win_len, hop_len: in samples (not seconds).
    """
    
    # load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    rows = []
    
    for seg_idx, seg in enumerate(segments):
        seg_start = int(seg["start_s"] * sr)
        seg_end   = int(seg["end_s"] * sr)
        label     = seg["label"]
        
        # Walk through segment with sliding windows
        for start in range(seg_start, seg_end - win_len + 1, hop_len):
            end = start + win_len
            chunk = y[start:end]

            # Check chunk is not pure silence
            if np.all(chunk < 1e-4):
                continue

            feats = compute_audio_features(chunk, sr, feature_list)
            
            row = {
                "audio_path": audio_path,
                "segment_idx": seg_idx,
                "segment_label": label,
                "window_start_s": start / sr,
                "window_end_s": end / sr
            }
            for feat_name, feat in feats.items():
                row[feat_name] = feat
            rows.append(row)
            break
        break
    
    return pd.DataFrame(rows)


def build_master_table(items, feature_list, win_len=2048, hop_len=512, sr=22050):
    """
    items: list of dicts like {"audio_path": ".../file.wav", "segments": [...]}
    Returns a single DataFrame with all windows from all files.
    Saves features to npy/npz and metadata to JSON.
    """
    dfs = []
    for audio_idx, item in enumerate(items):
        audio_path = item["audio_path"]
        segments   = item["segments"]

        df_file = make_windowed_data(
            audio_path=audio_path,
            segments=segments,
            feature_list=feature_list,
            win_len=win_len,
            hop_len=hop_len,
            sr=sr
        )

        # add identifiers & housekeeping
        audio_id = Path(audio_path).stem
        df_file["audio_id"] = audio_id
        df_file["file_path"] = str(audio_path)
        df_file["duration_s"] = df_file["window_end_s"] - df_file["window_start_s"]
        df_file["sr"] = sr
        df_file["win_len"] = win_len
        df_file["hop_len"] = hop_len

        dfs.append(df_file)
        break

    master = pd.concat(dfs, ignore_index=True)
    master.insert(0, "window_id", master.index.map(lambda i: f"w{i:010d}"))

    # Expand feature vector columns (e.g., CLAP) into separate columns
    expanded = master.copy()
    for col in master.columns:
        if master[col].dtype == 'object' and isinstance(master[col].iloc[0], (np.ndarray, list)):
            arrs = np.stack(master[col].values)
            for i in range(arrs.shape[1]):
                expanded[f"{col}_{i}"] = arrs[:, i]
            expanded = expanded.drop(columns=[col])

    # Select only numeric columns for npy/npz
    feature_cols = [col for col in expanded.columns if expanded[col].dtype in [np.float32, np.float64, np.int32, np.int64]]
    np.save('windowed_data.npy', expanded[feature_cols].values)
    np.savez('windowed_data.npz', **{col: expanded[col].values for col in feature_cols})

    # Save metadata (non-numeric columns and window_id)
    meta_cols = [col for col in expanded.columns if col not in feature_cols]
    meta_dict = expanded[meta_cols].to_dict(orient='list')
    feature_column_map = {}
    for feat in feature_list:
        if feat in expanded.columns:
            feature_column_map[feat] = [feat]
        else:
            # Vector features: collect all columns starting with feat + '_'
            cols = [col for col in expanded.columns if col.startswith(feat + '_')]
            feature_column_map[feat] = cols

    meta_dict['feature_column_map'] = feature_column_map
    import json
    with open('windowed_data_meta.json', 'w') as f:
        json.dump(meta_dict, f, indent=2)
    print("Saved features to windowed_data.npy and windowed_data.npz, metadata to windowed_data_meta.json")
    return expanded

if __name__ == "__main__":
    import json
    from argparse import ArgumentParser
    from pathlib import Path
    parser = ArgumentParser()
    parser.add_argument('--input_dir', default="../stopping_enhanced", type=str,
                        help="Directory containing metadata.json and audio files")
    parser.add_argument('--win_len', default=2048, type=int, help="Window length in samples")
    parser.add_argument('--hop_len', default=512, type=int, help="Hop length in samples")
    parser.add_argument('--sr', default=22050, type=int, help="Sample rate")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    audio_files = sorted(input_dir.glob("*.wav"))
    items = []
    for audio_path in audio_files:
        json_path = audio_path.with_name(audio_path.stem + "_enhanced.json")
        if not json_path.exists():
            print(f"Warning: No JSON file found for {audio_path}, skipping.")
            continue
        with open(json_path, 'r') as f:
            segments = json.load(f)
        items.append({
            "audio_path": str(audio_path),
            "segments": segments
        })

    feature_list = FEATURE_NAME_MAP.keys()

    master_df = build_master_table(
        items=items,
        feature_list=feature_list,
        win_len=args.win_len,
        hop_len=args.hop_len,
        sr=args.sr
    )

    print(f"Master table with {len(master_df)} windows saved to windowed_data.npy, windowed_data.npz, and windowed_data_meta.json")