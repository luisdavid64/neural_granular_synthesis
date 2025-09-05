import cv2
import json
import time
import sys
import pathlib
import numpy as np
import sounddevice as sd
from moviepy.video.io.VideoFileClip import VideoFileClip

# -------- settings --------
VIDEO_GLOB = "*.mp4"
TARGET_PRESSES = 3       # set to any number; press 'q' to finish early
ROUND_DECIMALS = 3       # seconds precision in JSON
WINDOW_NAME = "Labeler (SPACE=mark, Backspace=undo, q=next)"
# --------------------------

def play_video_and_mark(path: pathlib.Path):
    print(f"\n=== {path.name} ===")
    # ---- open video for frames ----
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("  Could not open video with OpenCV. Skipping.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    frame_period = 1.0 / fps

    # ---- extract audio via moviepy ----
    clip = VideoFileClip(str(path))
    if clip.audio is None:
        print("  (No audio track) Showing silent video.")
        sr = None
        audio = None
        duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    else:
        sr = int(clip.audio.fps)
        audio = clip.audio.to_soundarray(fps=sr).astype(np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        duration = clip.duration

    # small fade to avoid clicks
    if audio is not None:
        fade = min(1024, len(audio))
        if fade > 0:
            w = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
            audio[:fade] *= w
            audio[-fade:] *= w[::-1]

    marks, done = [], False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(WINDOW_NAME, 960, 540)  # uncomment if you want a fixed window size

    # start audio
    start_time = time.monotonic()
    if audio is not None:
        sd.play(audio, samplerate=sr, blocking=False)

    # display loop
    frame_idx = 0
    print("Press SPACE to mark (Backspace = undo last, q = next video).")
    try:
        while True:
            # Synchronize to elapsed time
            target_t = frame_idx * frame_period
            now = time.monotonic()
            elapsed = now - start_time
            if elapsed < target_t - 0.002:
                # sleep a hair to catch up to target time
                time.sleep(min(0.002, target_t - elapsed))
                continue

            ok, frame = cap.read()
            if not ok:
                # end of video
                break

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE
                t = round(time.monotonic() - start_time, ROUND_DECIMALS)
                # clamp to duration
                if duration is not None:
                    t = min(t, round(duration, ROUND_DECIMALS))
                marks.append(t)
                print(f"  mark {len(marks)}/{TARGET_PRESSES}: {t} s")
                if len(marks) >= TARGET_PRESSES:
                    done = True
                    break

            elif key == 8:  # Backspace
                if marks:
                    removed = marks.pop()
                    print(f"  undo → removed {removed} s (now {len(marks)}/{TARGET_PRESSES})")

            elif key == ord('q'):  # quit/next
                done = True
                break
                
            elif key == ord('r'): # restart video and reset marks
                print("  restarting video and resetting marks")
                marks = []
                frame_idx = 0
                start_time = time.monotonic()
                if audio is not None:
                    sd.stop()
                    sd.play(audio, samplerate=sr, blocking=False)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # esc quit
            elif key == 27:
                print("  (Exiting entirely.)")
                sys.exit(0)

            frame_idx += 1

            # Stop if audio finished (in case of audio-only short/long)
            if duration is not None and (time.monotonic() - start_time) >= duration:
                break

    finally:
        cap.release()
        cv2.destroyWindow(WINDOW_NAME)
        sd.stop()
        clip.close()

    return marks

def main():
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
    files = sorted(root.glob(VIDEO_GLOB))
    if not files:
        print(f"No {VIDEO_GLOB} files found in {root.resolve()}")
        sys.exit(1)

    for p in files:
        marks = play_video_and_mark(p)
        out = p.with_suffix(".json")
        out.write_text(json.dumps(marks, indent=2))
        print(f"Saved {out.name}: {marks}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
