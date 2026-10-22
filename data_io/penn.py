"""
data_io/penn.py

Penn Action loading helpers and clip selection for the experiments.

- load_labels():   GT keypoints, visibility mask, bbox and metadata of one clip
- frame_paths():   sorted frame image paths of one clip
- penn_to_coco():  put Penn joints 1..12 into COCO-17 slots 5..16
- chain_visibility(), select_clips():  pick the squat test / tuning clips

Penn Action joint order (13 joints, 0-based):
  0 head, 1 L_shoulder, 2 R_shoulder, 3 L_elbow, 4 R_elbow, 5 L_wrist,
  6 R_wrist, 7 L_hip, 8 R_hip, 9 L_knee, 10 R_knee, 11 L_ankle, 12 R_ankle
Penn joints 1..12 == COCO keypoints 5..16 (same order).

Invisible joints carry junk coordinates: GT is only valid where vis is True.

Usage (prints the selected clip lists and saves them as CSV):
  python -m data_io.penn
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

PENN_ROOT = Path("data/Penn_Action")
INDEX_CSV = Path("results/penn_index.csv")

JOINT_NAMES = [
    "head", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist",
    "R_wrist", "L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle",
]
N_PENN = 13

# Same-side leg chains (Penn indices): shoulder, hip, knee, ankle
LEG_CHAIN = {
    "L": [1, 7, 9, 11],
    "R": [2, 8, 10, 12],
}

# Penn joints 1..12  <->  COCO keypoints 5..16
PENN_BODY = slice(1, 13)
COCO_BODY = slice(5, 17)
N_COCO = 17


def clip_id(clip) -> str:
    """Normalise a clip id (int, '1', '0001') to the zero-padded string '0001'."""
    return str(clip).zfill(4)


def load_labels(clip, root: Path = PENN_ROOT) -> dict:
    """Load the Penn Action annotation of one clip.   [glue: filled in]

    Args:
        clip: clip id (any form accepted by clip_id()).
        root: Penn Action root containing labels/ and frames/.

    Returns dict with:
        clip    str          zero-padded id
        action  str          e.g. "squat"
        view    str          "front" | "back" | "left" | "right"
        train   int          1 (train) or -1 (test)
        T       int          number of frames
        H, W    int          frame height / width in px
        kp      (T, 13, 2)   float64 GT pixel coords (x, y); junk where not vis
        vis     (T, 13)      bool, True where the joint is annotated visible
        bbox    (T, 4)       float64 [x1, y1, x2, y2] person box in px
    """
    m = sio.loadmat(Path(root) / "labels" / f"{clip_id(clip)}.mat", squeeze_me=True)
    x = np.atleast_2d(np.asarray(m["x"], dtype=float))
    y = np.atleast_2d(np.asarray(m["y"], dtype=float))
    vis = np.atleast_2d(np.asarray(m["visibility"])) == 1
    T = int(m["nframes"])
    dims = np.atleast_1d(m["dimensions"]).astype(int)    # [H, W, nframes]

    # Some clips (e.g. 1865) have one bbox row fewer than frames: repeat the last row
    bbox = np.atleast_2d(np.asarray(m["bbox"], dtype=float))
    if bbox.shape[0] < T:
        bbox = np.vstack([bbox, np.repeat(bbox[-1:], T - bbox.shape[0], axis=0)])

    return {
        "clip": clip_id(clip),
        "action": str(m["action"]),
        "view": str(m["pose"]),
        "train": int(m["train"]),
        "T": T,
        "H": int(dims[0]),
        "W": int(dims[1]),
        "kp": np.stack([x, y], axis=-1),
        "vis": vis,
        "bbox": bbox[:T],
    }


def frame_paths(clip, root: Path = PENN_ROOT) -> list[Path]:
    """Sorted list of frame image paths of one clip (000001.jpg, ...).   [glue: filled in]"""
    paths = sorted((Path(root) / "frames" / clip_id(clip)).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frames for clip {clip_id(clip)} under {root}/frames")
    return paths


def penn_to_coco(kp: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place Penn body joints into COCO-17 slots so GT and YOLO arrays line up.   [glue: filled in]

    Penn head has no COCO equivalent (COCO 0..4 are nose/eyes/ears), so
    COCO slots 0..4 are NaN and marked not visible.

    Args:
        kp:  (T, 13, 2) Penn keypoints.
        vis: (T, 13) bool visibility.

    Returns:
        kp17:  (T, 17, 2) float, NaN in slots 0..4
        vis17: (T, 17) bool, False in slots 0..4
    """
    T = kp.shape[0]
    kp17 = np.full((T, N_COCO, 2), np.nan)
    vis17 = np.zeros((T, N_COCO), dtype=bool)
    kp17[:, COCO_BODY] = kp[:, PENN_BODY]
    vis17[:, COCO_BODY] = vis[:, PENN_BODY]
    return kp17, vis17


def chain_visibility(vis: np.ndarray, chain: list[int]) -> float:
    """Fraction of frames in which ALL joints of `chain` are visible.

    Args:
        vis:   (T, 13) bool visibility.
        chain: Penn joint indices, e.g. LEG_CHAIN["L"] = [1, 7, 9, 11].

    Returns:
        float in [0, 1].
    """
    return float(vis[:, chain].all(axis=1).mean())


def select_clips(split: int, action: str = "squat",
                 views: tuple[str, ...] = ("left", "right"),
                 min_chain_vis: float = 0.8, max_clips: int | None = 40,
                 index_csv: Path = INDEX_CSV, root: Path = PENN_ROOT) -> pd.DataFrame:
    """Select clips for an experiment set.

    A clip is kept if action matches, view is in `views`, train == split, and
    at least one same-side leg chain (shoulder, hip, knee, ankle) is fully
    visible in >= min_chain_vis of its frames.

    Args:
        split:         -1 for the test set, 1 for the tuning (train) set.
        action, views: filters on the index columns.
        min_chain_vis: threshold on max(vis_L, vis_R).
        max_clips:     keep at most this many (first ones by clip id), None = all.

    Returns:
        DataFrame, one row per kept clip, sorted by clip id, columns:
          clip (str, zero-padded), view, nframes, vis_L, vis_R,
          side ("L" or "R": the better-visible chain, used for near-side plots)

    Expected (checked once): split=-1 -> 37 clips, split=1 -> 46 clips (before the cap).
    """
    df = pd.read_csv(index_csv, dtype={"clip": str})
    df["clip"] = df["clip"].str.zfill(4)
    
    mask = (df["action"]==action)&(df["view"].isin(views))&(df["train"]==split)
    sub = df[mask]
    
    rows = []
    for _, row in sub.iterrows():
        lab = load_labels(row['clip'], root)
        vis_L = chain_visibility(lab["vis"], LEG_CHAIN["L"])
        vis_R = chain_visibility(lab["vis"], LEG_CHAIN["R"])
        rows.append({
            "clip": row["clip"],
            "view": row["view"],
            "nframes": row["nframes"],
            "vis_L": vis_L,
            "vis_R": vis_R,
            "side": "L" if vis_L >= vis_R else "R"
        })
    out = pd.DataFrame(rows)
    out = out[out[["vis_L", "vis_R"]].max(axis=1) >= min_chain_vis]
    out = out.sort_values("clip")
    if max_clips is not None:
        out = out.head(max_clips)
    
    return out.reset_index(drop=True)


def main():
    """CLI: print the test / tuning clip lists and save them.   [glue: filled in]"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--n-tune", type=int, default=10)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    sets = {
        "test": select_clips(split=-1, max_clips=args.n_test),
        "tune": select_clips(split=1, max_clips=args.n_tune),
    }
    for name, sel in sets.items():
        out = Path(args.out_dir) / f"clips_{name}.csv"
        sel.to_csv(out, index=False)
        print(f"--- {name}: {len(sel)} clips -> {out} ---")
        print(sel.round(2).to_string(index=False), "\n")
        print("views:", sel["view"].value_counts().to_dict(),
              "| frames total:", int(sel["nframes"].sum()), "\n")

    # Sanity check of the loader on the first test clip
    lab = load_labels(sets["test"]["clip"].iloc[0])
    print("load_labels check:", lab["clip"], lab["view"], "kp", lab["kp"].shape,
          "vis", lab["vis"].shape, "bbox", lab["bbox"].shape,
          "frames", len(frame_paths(lab["clip"])))


if __name__ == "__main__":
    main()
