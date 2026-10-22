"""
tools/inspect_penn.py

Index the Penn Action dataset: one row per clip with action, camera view,
frame count, frame size and keypoint visibility. Prints a summary and saves
results/penn_index.csv so we can pick clips for the experiments.

Penn Action joint order (13 joints, 0-based):
  0 head, 1 L_shoulder, 2 R_shoulder, 3 L_elbow, 4 R_elbow, 5 L_wrist,
  6 R_wrist, 7 L_hip, 8 R_hip, 9 L_knee, 10 R_knee, 11 L_ankle, 12 R_ankle
Joints 1..12 are in exactly the same order as COCO keypoints 5..16,
so  penn[:, 1:13]  <->  coco[:, 5:17].

Usage:
  python tools/inspect_penn.py --root data/Penn_Action
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def load_clip(mat_path: Path) -> dict:
    """Load one Penn Action label file into a flat dict."""
    m = sio.loadmat(mat_path, squeeze_me=True)
    vis = np.asarray(m["visibility"], dtype=float)      # (nframes, 13)
    dims = np.atleast_1d(m["dimensions"]).astype(int)    # [H, W, nframes]
    return {
        "clip": mat_path.stem,
        "action": str(m["action"]),
        "view": str(m["pose"]),
        "nframes": int(m["nframes"]),
        "height": int(dims[0]),
        "width": int(dims[1]),
        "train": int(m["train"]),
        # mean visibility of the 12 body joints (skip head)
        "body_vis": float(vis[:, 1:13].mean()),
        # fraction of frames where at least one body joint is occluded
        "occl_frames": float((vis[:, 1:13] < 0.5).any(axis=1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Penn_Action")
    ap.add_argument("--out", default="results/penn_index.csv")
    args = ap.parse_args()

    label_dir = Path(args.root) / "labels"
    mats = sorted(label_dir.glob("*.mat"))
    if not mats:
        raise SystemExit(f"No .mat files found in {label_dir}")

    # Show the raw fields of the first file once, to confirm the format
    first = sio.loadmat(mats[0], squeeze_me=True)
    print("Fields in", mats[0].name, ":",
          [k for k in first.keys() if not k.startswith("__")])

    df = pd.DataFrame([load_clip(p) for p in mats])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n{len(df)} clips indexed -> {args.out}\n")
    print("Clips per action and view:")
    print(pd.crosstab(df["action"], df["view"]), "\n")

    for action in ["squat", "jumping_jacks"]:
        sub = df[df["action"] == action]
        print(f"--- {action}: {len(sub)} clips ---")
        print(sub.groupby("view")[["nframes", "height", "width", "body_vis", "occl_frames"]]
              .mean().round(2), "\n")


if __name__ == "__main__":
    main()
