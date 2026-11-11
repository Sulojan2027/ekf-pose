"""
Index the Penn Action dataset into results/penn_index.csv.

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
        "body_vis": float(vis[:, 1:13].mean()),
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
