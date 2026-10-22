"""
pose/run_pose.py

Run YOLO26-pose on Penn Action frame folders and cache the raw keypoints.
Ultralytics is used ONLY for model.predict(); no tracking or smoothing.

For every clip and model, saves results/keypoints/<clip>_<model>.npz with:
  kp         (T, 17, 3)  float32  COCO-17 keypoints (x, y, conf) of the person
                                   with the largest box; zeros if nobody detected
  box        (T, 4)      float32  that person's box [x1, y1, x2, y2] (zeros if none)
  box_conf   (T,)        float32  that person's box confidence (0 if none)
  n_persons  (T,)        int16    number of detections in the frame
  t_infer_ms (T,)        float32  Ultralytics model forward time per frame (ms)
  t_total_ms (T,)        float32  preprocess + inference + postprocess (ms)
  model, imgsz                     metadata

Existing cache files are skipped, so the script can be stopped and restarted.

Usage (run in a second terminal, it takes a while):
  python -m pose.run_pose                         # nano + x, test + tune clips
  python -m pose.run_pose --models yolo26n --sets test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from data_io.penn import clip_id, frame_paths

WEIGHTS = {
    "yolo26n": Path("weights/yolo26n-pose.pt"),
    "yolo26x": Path("weights/yolo26x-pose.pt"),
}
OUT_DIR = Path("results/keypoints")
N_KP = 17


def cache_path(clip, model_name: str, out_dir: Path = OUT_DIR) -> Path:
    """Path of the keypoint cache file of one clip and model."""
    return Path(out_dir) / f"{clip_id(clip)}_{model_name}.npz"


def load_keypoints(clip, model_name: str = "yolo26n", out_dir: Path = OUT_DIR) -> dict:
    """Load a cached keypoint file as a plain dict of arrays."""
    with np.load(cache_path(clip, model_name, out_dir)) as f:
        return {k: f[k] for k in f.files}


def run_clip(model: YOLO, clip, imgsz: int, device) -> dict:
    """Run pose estimation on every frame of one clip; keep the largest person.

    Args:
        model:  loaded YOLO pose model.
        clip:   clip id.
        imgsz:  inference size (letterboxed), 640.
        device: CUDA device index (0) or "cpu".

    Returns:
        dict with the arrays listed in the module docstring (kp, box, box_conf,
        n_persons, t_infer_ms, t_total_ms), each with first dim T = #frames.
        Frames with no detection keep the zero defaults (kp conf = 0), so the
        EKF later treats all their keypoints as invisible and just predicts.
    """
    paths = frame_paths(clip)
    T = len(paths)
    out = {
        "kp": np.zeros((T, N_KP, 3), np.float32),
        "box": np.zeros((T, 4), np.float32),
        "box_conf": np.zeros(T, np.float32),
        "n_persons": np.zeros(T, np.int16),
        "t_infer_ms": np.zeros(T, np.float32),
        "t_total_ms": np.zeros(T, np.float32),
    }
    for t, p in enumerate(paths):
        r = model.predict(str(p), imgsz=imgsz, device=device, verbose=False)[0]
        
        out["t_infer_ms"][t] = r.speed.get("inference")
        out["t_total_ms"][t] = sum(r.speed.values())
        
        n = len(r.boxes) if r.boxes is not None else 0
        if n != 0:
            out["n_persons"][t] = n
        else:
            continue
        
        xyxy = r.boxes.xyxy.cpu().numpy()
        area = [(x2 - x1) * (y2 - y1) for [x1,y1,x2,y2] in xyxy]
        i = int(np.argmax(area))
        
        out["kp"][t] = r.keypoints.data[i].cpu().numpy()
        out["box"][t] = xyxy[i]
        out["box_conf"][t] = float(r.boxes.conf[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo26n", "yolo26x"])
    ap.add_argument("--sets", nargs="+", default=["test", "tune"])
    ap.add_argument("--clips", nargs="*", help="explicit clip ids (overrides --sets)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    if args.clips:
        clips = [clip_id(c) for c in args.clips]
    else:
        clips = []
        for s in args.sets:
            sel = pd.read_csv(f"results/clips_{s}.csv", dtype={"clip": str})
            clips += [clip_id(c) for c in sel["clip"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        todo = [c for c in clips if not cache_path(c, name).exists()]
        print(f"{name}: {len(clips) - len(todo)} cached, {len(todo)} to run")
        if not todo:
            continue
        model = YOLO(str(WEIGHTS[name]))
        # warm-up so CUDA init / first-call overhead doesn't pollute the timings
        model.predict(str(frame_paths(todo[0])[0]), imgsz=args.imgsz,
                      device=args.device, verbose=False)

        for c in tqdm(todo, desc=name):
            res = run_clip(model, c, args.imgsz, args.device)
            np.savez_compressed(cache_path(c, name), model=name, imgsz=args.imgsz, **res)

        print(f"{name}: done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
