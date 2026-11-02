"""
experiments/exp_real.py

Real-data experiments on Penn Action (GT = Penn labels, already L/R-mirror-fixed in data_io).

  tune : grid search on the 10 TUNING clips only (never the test clips)
         EKF sigma_alpha x sigma0, moving-average window, per-keypoint-KF sigma_a
         -> results/tuned_params.json, results/tables/e2_real_tune.csv, figure e2_real_tune
  run  : everything below on the 37 TEST clips with the tuned parameters
         E6  raw vs moving average vs per-keypoint KF vs EKF (yolo26n)   -> e6_methods (+ e6_trace fig)
         E5  EKF ablation: adaptive R / gating / joint limits / re-init / far leg -> e5_real_ablation
         E7  yolo26n(+EKF) vs yolo26x(+EKF): accuracy vs runtime           -> e7_edge (+ fig)
         E8  bone lengths over time, one clip                              -> e8_bone_lengths fig
         E10 overlay of GT / raw / EKF skeletons on frames, one clip       -> e10_overlay fig

Scoring rules (fairness, write them in the report):
  - every method uses the same cached YOLO keypoints and the same c_min;
  - angle RMSE / lag are computed on the frames where GT is visible AND every compared
    method has an estimate (common frames); `coverage` = share of GT-visible frames where
    the method has an estimate at all (raw has gaps, the EKF coasts through them);
  - numbers are per-clip metrics averaged over clips (+ medians); per-clip values are saved too.
  - "EKF, far leg ignored": YOLO is ~95 % confident on the SELF-OCCLUDED far-side leg in side
    views, and the EKF's shared bone lengths let those guesses corrupt the near leg. This variant
    drops the far knee + ankle. The far side comes from Penn's `view` label (camera metadata,
    matches the near side in 46/47 clips), NOT from GT keypoints, so it is a fair input.

Usage:
  python -m experiments.exp_real tune
  python -m experiments.exp_real run                 # all of E5 E6 E7 E8 E10
  python -m experiments.exp_real run --only E6 E7
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines.kf_per_keypoint import kf_per_keypoint
from baselines.moving_average import moving_average
from data_io.penn import frame_paths, load_labels
from ekf import skeleton as sk
from ekf.ekf import PoseEKF
from ekf.models import EKFParams
from eval import metrics as M

FIG, TAB = Path("report/figures"), Path("results/tables")
KP_DIR, TUNED = Path("results/keypoints"), Path("results/tuned_params.json")
COL = {"GT": "k", "raw": "#9ca3af", "moving average": "#16a34a", "per-keypoint KF": "#ea580c", "EKF": "#2563eb",
       "EKF, far leg ignored": "#7c3aed"}
FAR_LEG = {"L": [sk.R_KN, sk.R_AN], "R": [sk.L_KN, sk.L_AN]}       # far side = opposite of the view

# tuning grids
GRID_ALPHA = [0.003, 0.006, 0.01, 0.02, 0.03, 0.05]
GRID_S0 = [3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
GRID_MA = [3, 5, 7, 9]
GRID_KF = [0.5, 1.0, 2.0, 4.0, 8.0]


# ----------------------------------------------------------------------------- data
def clip_list(name: str) -> pd.DataFrame:
    df = pd.read_csv(f"results/clips_{name}.csv", dtype={"clip": str})
    df["clip"] = df["clip"].str.zfill(4)
    return df


def load_clip(clip: str, side: str, model: str = "yolo26n", view: str = "") -> dict:
    """GT + cached YOLO keypoints of one clip, both in local order 0..11 (COCO 5..16)."""
    lab = load_labels(clip)
    f = np.load(KP_DIR / f"{clip}_{model}.npz")
    T = min(lab["T"], len(f["kp"]))
    kp, conf = sk.split_coco(f["kp"][:T].astype(float))
    return {"clip": clip, "side": side, "view": view, "T": T, "gt": lab["kp"][:T, 1:13], "vis": lab["vis"][:T, 1:13],
            "bbox": lab["bbox"][:T], "kp": kp, "conf": conf, "t_infer": float(np.mean(f["t_infer_ms"][:T]))}


def load_set(name: str, model: str = "yolo26n") -> list[dict]:
    return [load_clip(r["clip"], r["side"], model, r["view"]) for _, r in clip_list(name).iterrows()]


def drop_far_leg(d: dict) -> dict:
    """Copy of the clip with the far-side knee + ankle confidences set to 0 (from the view label)."""
    dd = dict(d)
    dd["conf"] = d["conf"].copy()
    dd["conf"][:, FAR_LEG[d["view"][0].upper()]] = 0.0
    return dd


# ----------------------------------------------------------------------------- methods
def run_ekf(d, p: EKFParams) -> dict:
    out = PoseEKF(p).run(d["kp"], d["conf"])
    used, gated = out["n_used"].sum(), out["n_gated"].sum()
    return {"kp": out["z"].reshape(-1, 12, 2), "t_ms": float(out["t_ms"].mean()), "out": out,
            "gated_pct": 100.0 * gated / max(used + gated, 1), "reinits": int(out["n_reinit"].sum()),
            "nis_per_dof": float(np.nansum(out["nis"]) / max(out["dof"].sum(), 1))}   # ~1 = honest R


def run_method(name: str, d: dict, cfg: dict) -> dict:
    """One method on one clip -> {"kp": (T, 12, 2), "t_ms": filter ms/frame, ...}."""
    p = dataclasses.replace(EKFParams(), **cfg["ekf"])
    t0 = time.perf_counter()
    if name == "raw":
        kp = M.mask_low_conf(d["kp"], d["conf"], p.c_min)
    elif name == "moving average":
        kp = moving_average(d["kp"], d["conf"], cfg["ma_window"], p.c_min)
    elif name == "per-keypoint KF":
        kp = kf_per_keypoint(d["kp"], d["conf"], p, sigma_a=cfg["kf_sigma_a"])[0]
    else:
        raise ValueError(name)
    return {"kp": kp, "t_ms": (time.perf_counter() - t0) * 1e3 / d["T"]}


def score(d: dict, kps: dict) -> list[dict]:
    """Per-method metric rows for one clip (common frames across the methods in `kps`)."""
    side = d["side"]
    knee_ok, hip_ok = M.angle_valid(d["vis"], side)
    gk, gh = M.joint_angles(d["gt"], side)
    ang = {m: M.joint_angles(v["kp"], side) for m, v in kps.items()}
    ck = knee_ok & np.all([np.isfinite(a[0]) for a in ang.values()], 0)
    ch = hip_ok & np.all([np.isfinite(a[1]) for a in ang.values()], 0)
    rows = [{"clip": d["clip"], "method": "GT", "knee_jitter": M.jitter(np.where(knee_ok, gk, np.nan)),
             "hip_jitter": M.jitter(np.where(hip_ok, gh, np.nan)), "bone_cv": M.bone_length_cv(d["gt"], side, knee_ok)}]
    for m, v in kps.items():
        k, h = ang[m]
        rows.append({
            "clip": d["clip"], "method": m,
            "knee_rmse": M.angle_rmse(k, gk, ck), "hip_rmse": M.angle_rmse(h, gh, ch),
            "knee_jitter": M.jitter(k), "hip_jitter": M.jitter(h),
            "knee_lag": M.lag(k, gk, ck), "bone_cv": M.bone_length_cv(v["kp"], side, knee_ok),
            "kp_rmse_pct": M.keypoint_rmse_pct(v["kp"], d["gt"], d["vis"], d["bbox"]),
            "coverage": 100.0 * np.isfinite(k)[knee_ok].mean() if knee_ok.any() else np.nan,
            "filter_ms": v.get("t_ms", np.nan), "gated_pct": v.get("gated_pct", np.nan),
            "reinits": v.get("reinits", np.nan), "nis_per_dof": v.get("nis_per_dof", np.nan),
        })
    return rows


def summarise(rows: list[dict], name: str, order: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}_per_clip.csv", index=False)
    keep = [m for m in order if m in set(df["method"])]
    s = df.groupby("method").mean(numeric_only=True).reindex(keep)
    med = df.groupby("method")[["knee_rmse", "hip_rmse"]].median().reindex(keep) if "knee_rmse" in df else None
    if med is not None:
        s.insert(0, "knee_rmse_median", med["knee_rmse"]); s.insert(1, "hip_rmse_median", med["hip_rmse"])
    s = s.dropna(axis=1, how="all").reset_index()
    s.to_csv(TAB / f"{name}.csv", index=False)
    print(f"  table -> {TAB / name}.csv  (per clip: {name}_per_clip.csv)\n{s.round(2).to_string(index=False)}\n")
    return s


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {FIG / name}.png")


# ----------------------------------------------------------------------------- tune
def objective(rows, method):
    """MEDIAN over clips of the per-clip mean(knee RMSE, hip RMSE): a few clips with bad GT /
    extreme foreshortening would otherwise decide the parameters on only 10 tuning clips."""
    df = pd.DataFrame([r for r in rows if r["method"] == method])
    return float(np.nanmedian(df[["knee_rmse", "hip_rmse"]].mean(axis=1)))


def tune():
    data = load_set("tune")
    print(f"tuning on {len(data)} tuning clips (objective = median over clips of mean(knee, hip) RMSE)")
    base = {"ekf": {}, "ma_window": 5, "kf_sigma_a": 2.0}
    grid = []
    for sa in GRID_ALPHA:
        for s0 in GRID_S0:
            p = EKFParams(sigma_alpha=sa, sigma0=s0)
            rows = sum((score(d, {"EKF": run_ekf(d, p)}) for d in data), [])
            df = pd.DataFrame([r for r in rows if r["method"] == "EKF"])
            grid.append({"sigma_alpha": sa, "sigma0": s0, "objective": objective(rows, "EKF"),
                         "knee_rmse": df["knee_rmse"].mean(), "hip_rmse": df["hip_rmse"].mean(),
                         "knee_jitter": df["knee_jitter"].mean()})
    g = pd.DataFrame(grid)
    best = g.loc[g["objective"].idxmin()]
    base["ekf"] = {"sigma_alpha": float(best["sigma_alpha"]), "sigma0": float(best["sigma0"])}  # KF shares R
    ma = {w: objective(sum((score(d, {"m": run_method("moving average", d, {**base, "ma_window": w})})
                            for d in data), []), "m") for w in GRID_MA}
    kf = {a: objective(sum((score(d, {"m": run_method("per-keypoint KF", d, {**base, "kf_sigma_a": a})})
                            for d in data), []), "m") for a in GRID_KF}
    cfg = {"ekf": base["ekf"],
           "ma_window": int(min(ma, key=ma.get)), "kf_sigma_a": float(min(kf, key=kf.get))}
    TUNED.write_text(json.dumps(cfg, indent=2))
    TAB.mkdir(parents=True, exist_ok=True)
    g.to_csv(TAB / "e2_real_tune.csv", index=False)
    print(g.round(3).to_string(index=False))
    print("moving average:", {k: round(v, 3) for k, v in ma.items()})
    print("per-keypoint KF:", {k: round(v, 3) for k, v in kf.items()})
    print(f"-> {TUNED}: {cfg}")

    G = g.pivot(index="sigma_alpha", columns="sigma0", values="objective").values
    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(G, cmap="viridis_r", origin="lower")
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ax.text(j, i, f"{G[i, j]:.1f}", ha="center", va="center", color="w", fontsize=7)
    ax.set_xticks(range(len(GRID_S0)), [str(v) for v in GRID_S0]); ax.set_yticks(range(len(GRID_ALPHA)), [str(v) for v in GRID_ALPHA])
    ax.set_xlabel("σ0 (px)"); ax.set_ylabel("σ_α (rad/frame²)"); fig.colorbar(im, ax=ax, fraction=0.046, label="median clip RMSE, knee/hip mean (deg)")
    ax.set_title("E2 (real, tuning clips): EKF parameter grid", fontsize=10)
    savefig(fig, "e2_real_tune")


def load_tuned() -> dict:
    if not TUNED.exists():
        print(f"WARNING: {TUNED} missing, using defaults. Run `python -m experiments.exp_real tune` first.")
        return {"ekf": {}, "ma_window": 5, "kf_sigma_a": 2.0}
    return json.loads(TUNED.read_text())


# ----------------------------------------------------------------------------- E6 / E5 / E7
def e6(cfg, data):
    print("E6: methods on the test set (yolo26n)")
    p = dataclasses.replace(EKFParams(), **cfg["ekf"])
    rows, cache = [], {}
    for d in data:
        kps = {m: run_method(m, d, cfg) for m in ("raw", "moving average", "per-keypoint KF")}
        kps["EKF"] = run_ekf(d, p)
        kps["EKF, far leg ignored"] = run_ekf(drop_far_leg(d), p)
        cache[d["clip"]] = kps
        rows += score(d, kps)
    summarise(rows, "e6_methods", ["GT", "raw", "moving average", "per-keypoint KF", "EKF", "EKF, far leg ignored"])
    trace_figure(data, cache)
    return cache


def trace_figure(data, cache):
    """Knee angle of the clip with the most GT-visible frames: every method vs GT."""
    d = max(data, key=lambda d: M.angle_valid(d["vis"], d["side"])[0].sum())
    kps, side = cache[d["clip"]], d["side"]
    t = np.arange(d["T"])
    ok = M.angle_valid(d["vis"], side)[0]
    gk = np.degrees(np.where(ok, M.joint_angles(d["gt"], side)[0], np.nan))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t, gk, color="k", lw=2, label="Penn GT")
    for m in ("raw", "moving average", "per-keypoint KF"):
        ax.plot(t, np.degrees(M.joint_angles(kps[m]["kp"], side)[0]), color=COL[m], lw=1.2,
                ls="-" if m != "raw" else ":", label=m)
    out = kps["EKF"]["out"]
    th, sh = (sk.ANG_THIGH_L, sk.ANG_SHANK_L) if side == "L" else (sk.ANG_THIGH_R, sk.ANG_SHANK_R)
    s = np.degrees(np.sqrt(out["P"][:, th, th] + out["P"][:, sh, sh] - 2 * out["P"][:, th, sh]))
    ke = np.degrees(M.joint_angles(kps["EKF"]["kp"], side)[0])
    ax.fill_between(t, ke - 2 * s, ke + 2 * s, color=COL["EKF"], alpha=0.18)
    ax.plot(t, ke, color=COL["EKF"], lw=2, label="EKF ± 2σ")
    ax.plot(t, np.degrees(M.joint_angles(kps["EKF, far leg ignored"]["kp"], side)[0]), color=COL["EKF, far leg ignored"],
            lw=1.5, label="EKF, far leg ignored")
    ax.set_xlabel("frame"); ax.set_ylabel(f"knee flexion, {side} leg (deg)"); ax.legend(fontsize=8, ncol=3, frameon=False)
    ax.set_title(f"E6: clip {d['clip']} ({d['T']} frames, yolo26n keypoints)", fontsize=10)
    savefig(fig, "e6_trace")


def e5(cfg, data):
    print("E5 (real): EKF ablation on the test set")
    base = dict(cfg["ekf"])
    variants = {"EKF (full)": {}, "EKF fixed R": {"adaptive_R": False}, "EKF no gating": {"gating": False},
                "EKF no joint limits": {"joint_limits": False},
                "EKF no re-init": {"reinit_after": 0},
                "EKF fixed R + no gating": {"adaptive_R": False, "gating": False}}
    rows = []
    for d in data:
        kps = {n: run_ekf(d, dataclasses.replace(EKFParams(), **{**base, **ov})) for n, ov in variants.items()}
        kps["EKF, far leg ignored"] = run_ekf(drop_far_leg(d), dataclasses.replace(EKFParams(), **base))
        rows += score(d, kps)
    summarise(rows, "e5_real_ablation", ["GT"] + list(variants) + ["EKF, far leg ignored"])


def e7(cfg, data_n):
    print("E7: edge-AI study, yolo26n vs yolo26x")
    p = dataclasses.replace(EKFParams(), **cfg["ekf"])
    data_x = {d["clip"]: d for d in load_set("test", "yolo26x")}
    rows = []
    for dn in data_n:
        dx = data_x[dn["clip"]]
        kps = {"yolo26n raw": run_method("raw", dn, cfg), "yolo26n + EKF": run_ekf(dn, p),
               "yolo26x raw": run_method("raw", dx, cfg), "yolo26x + EKF": run_ekf(dx, p)}
        for r in score(dn, kps):
            if r["method"] != "GT":
                r["yolo_ms"] = dn["t_infer"] if r["method"].startswith("yolo26n") else dx["t_infer"]
                r["total_ms"] = r["yolo_ms"] + (r["filter_ms"] if "EKF" in r["method"] else 0.0)
            rows.append(r)
    s = summarise(rows, "e7_edge", ["GT", "yolo26n raw", "yolo26n + EKF", "yolo26x raw", "yolo26x + EKF"])
    s = s[s["method"] != "GT"]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for _, r in s.iterrows():
        c = COL["EKF"] if "EKF" in r["method"] else COL["raw"]
        ax.scatter(r["total_ms"], r["knee_rmse"], s=70, color=c, marker="o" if "26n" in r["method"] else "s", zorder=3)
        ax.annotate(r["method"], (r["total_ms"], r["knee_rmse"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("time per frame (ms, log): YOLO forward + filter"); ax.set_ylabel("knee RMSE vs GT (deg)")
    ax.set_title("E7: accuracy vs runtime (RTX 2050)", fontsize=10); ax.grid(alpha=0.3)
    savefig(fig, "e7_edge")


# ----------------------------------------------------------------------------- E8 / E10
def showcase_clip(data):
    """Clip used for E8/E10: most GT-visible frames on the near-side leg."""
    return max(data, key=lambda d: M.angle_valid(d["vis"], d["side"])[0].sum())


def e8(cfg, data):
    print("E8: bone lengths over time")
    d = showcase_clip(data)
    side, t = d["side"], np.arange(d["T"])
    p = dataclasses.replace(EKFParams(), **cfg["ekf"])
    ekf = run_ekf(d, p)["out"]
    kf = run_method("per-keypoint KF", d, cfg)["kp"]
    raw = M.mask_low_conf(d["kp"], d["conf"], p.c_min)
    _, hp, kn, an = M.SIDE_KP[side]
    knee_ok = M.angle_valid(d["vis"], side)[0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (a, b, li, name) in zip(axs, [(hp, kn, sk.LEN_THIGH, "thigh"), (kn, an, sk.LEN_SHANK, "shank")]):
        L = lambda k: np.linalg.norm(k[:, b] - k[:, a], axis=-1)
        ax.plot(t, np.where(knee_ok, L(d["gt"]), np.nan), color="k", lw=1.5, label="from Penn GT")
        ax.plot(t, L(raw), color=COL["raw"], lw=1, label="from raw yolo26n")
        ax.plot(t, L(kf), color=COL["per-keypoint KF"], lw=1, label="from per-keypoint KF")
        m, s = ekf["x"][:, li], np.sqrt(ekf["P_diag"][:, li])
        ax.fill_between(t, m - 2 * s, m + 2 * s, color=COL["EKF"], alpha=0.2)
        ax.plot(t, m, color=COL["EKF"], lw=2, label="EKF state ± 2σ")
        ax.set_title(f"{name} length", fontsize=10); ax.set_xlabel("frame")
    axs[0].set_ylabel("px"); axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"E8: bone-length self-calibration, clip {d['clip']}", fontsize=11)
    savefig(fig, "e8_bone_lengths")


EDGES = [(sk.L_SH, sk.R_SH), (sk.L_HIP, sk.R_HIP), (sk.L_SH, sk.L_HIP), (sk.R_SH, sk.R_HIP),
         (sk.L_SH, sk.L_EL), (sk.L_EL, sk.L_WR), (sk.R_SH, sk.R_EL), (sk.R_EL, sk.R_WR),
         (sk.L_HIP, sk.L_KN), (sk.L_KN, sk.L_AN), (sk.R_HIP, sk.R_KN), (sk.R_KN, sk.R_AN)]


def draw(ax, kp, ok, color, lw, label=None):
    first = True
    for a, b in EDGES:
        if ok[a] and ok[b] and np.isfinite(kp[[a, b]]).all():
            ax.plot(kp[[a, b], 0], kp[[a, b], 1], color=color, lw=lw, label=label if first else None,
                    solid_capstyle="round")
            first = False


def e10(cfg, data):
    print("E10: overlay figure")
    d = showcase_clip(data)
    p = dataclasses.replace(EKFParams(), **cfg["ekf"])
    out = run_ekf(d, p)["out"]
    out_nf = run_ekf(drop_far_leg(d), p)["out"]
    knee = M.joint_angles(d["gt"], d["side"])[0]
    ok = M.angle_valid(d["vis"], d["side"])[0]
    deep = int(np.nanargmax(np.where(ok, knee, np.nan)))
    start = max(int(0.1 * d["T"]), 3)                      # 4 distinct phases: start, descent, bottom, ascent
    frames = sorted({start, (start + deep) // 2, deep, min(deep + int(0.6 * (d["T"] - 1 - deep)), d["T"] - 1)})
    paths = frame_paths(d["clip"])
    fig, axs = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 4))
    for ax, f in zip(np.atleast_1d(axs), frames):
        ax.imshow(cv2.cvtColor(cv2.imread(str(paths[f])), cv2.COLOR_BGR2RGB))
        draw(ax, d["gt"][f], d["vis"][f], "#22c55e", 4, "Penn GT")
        draw(ax, d["kp"][f], d["conf"][f] >= p.c_min, "#ef4444", 1.8, "raw yolo26n")
        draw(ax, out["z"][f].reshape(12, 2), np.ones(12, bool), "#3b82f6", 1.8, "EKF")
        draw(ax, out_nf["z"][f].reshape(12, 2), np.ones(12, bool), "#a855f7", 1.8, "EKF, far leg ignored")
        x1, y1, x2, y2 = d["bbox"][f]; m = 0.15 * (y2 - y1)
        ax.set_xlim(x1 - m, x2 + m); ax.set_ylim(y2 + m, y1 - m); ax.axis("off")
        ax.set_title(f"frame {f}", fontsize=9)
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"E10: clip {d['clip']}: GT (green), raw yolo26n (red), EKF (blue), EKF far leg ignored (purple)",
                 fontsize=11)
    savefig(fig, "e10_overlay")


RUN = {"E6": e6, "E5": e5, "E7": e7, "E8": e8, "E10": e10}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "run"])
    ap.add_argument("--only", nargs="*", default=list(RUN))
    a = ap.parse_args()
    if a.cmd == "tune":
        tune()
    else:
        cfg = load_tuned()
        print("parameters:", cfg)
        data = load_set("test")
        for name in a.only:
            RUN[name.upper()](cfg, data)
