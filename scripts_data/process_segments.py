
import argparse
import os
import json
import glob
import numpy as np 

def marks_to_segments(marks_s, duration_s, label_order=["silence","skin","fat","muscle"]):
    # label_order like ["skin","tissue","muscle","silence"]
    t = [0.0] + list(marks_s) + [duration_s]
    segs = []
    for i in range(len(t)-1):
        segs.append({"start_s": round(t[i], 3),
                     "end_s": round(t[i+1], 3),
                     "label": label_order[i] if i < len(label_order) else "unknown"})
    return segs

def get_labels_from_name(name):
    # Format: X1_X2_X3-thB-N.json, we want X1,X2,X3
    # e.g. S_Br_T
    # S = skin, Br = brain, F = fat, T = tissue, M = muscle
    abbrev_to_label = {
        "S": "skin",
        "Br": "brain",
        "F": "fat",
        "T": "tissue",
        "M": "muscle"
    }
    labels = []
    for part in name.split("_"):
        label = abbrev_to_label.get(part, "unknown")
        labels.append(label)
    return labels

def main(dir_path):
    # Go over json files in dir_path and create corresponding _enhanced.json files
    json_files = glob.glob(os.path.join(dir_path, "*.json"))
    for jf in json_files:
        with open(jf, 'r') as f:
            marks_s = json.load(f)
        # get duration from same named mp4 file
        mp4_file = jf[:-5] + ".mp4"
        if not os.path.exists(mp4_file):
            print(f"Warning: {mp4_file} not found, skipping {jf}")
            continue
        # get duration using cv2
        import cv2
        cap = cv2.VideoCapture(mp4_file)
        if not cap.isOpened():
            print(f"Warning: could not open {mp4_file}, skipping {jf}")
            continue
        duration_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        segments = marks_to_segments(marks_s, duration_s)
        cap.release()
        # Save as enhanced json
        enhanced_jf = jf[:-5] + "_enhanced.json"
        with open(enhanced_jf, 'w') as f:
            json.dump(segments, f, indent=2, sort_keys=True)
        print(f"Saved {enhanced_jf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', default="stopping_enhanced", type=str)
    args = parser.parse_args()
    main(args.dir_path)