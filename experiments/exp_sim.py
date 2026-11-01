"""
experiments/exp_sim.py

Simulation experiments (known ground truth):
  E1  knee/hip angle: truth vs raw vs EKF (+-2 sigma)        -> e1_sim_angles
  E2  3x3 sweep sigma_alpha x sigma0 (knee RMSE, jitter)      -> e2_sim_sweep
  E3  NEES / NIS vs chi^2 95 % band (model-matched truth)     -> e3_consistency     (NEVER DROP)
  E4  occlusion of knees+ankles for 10 / 20 / 30 frames        -> e4_occlusion
  E5s ablation (adaptive R, gating, joint limits) + baselines  -> e5_sim_ablation

Figures -> report/figures/<name>.png and .pdf ; tables -> results/tables/<name>.csv

Usage:
  python -m experiments.exp_sim                 # everything (~1 min)
  python -m experiments.exp_sim --only E1 E3
"""

import argparse
import dataclasses
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines.kf_per_keypoint import kf_per_keypoint
from baselines.moving_average import moving_average
from ekf import skeleton as sk
from ekf.ekf import PoseEKF
from ekf.models import EKFParams
from eval import metrics as M
from sim.simulate import LEG_KP, SimConfig, cv_states, observe, scenario, squat_states

FIG, TAB = Path("report/figures"), Path("results/tables")
C = {"truth": "k", "raw": "#9ca3af", "ma": "#16a34a", "kf": "#ea580c", "ekf": "#2563eb"}
WARM = 10                      # frames ignored at the start of every RMSE (filter warm-up)
SIDE = "L"                     # sim is symmetric; score the left leg


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {FIG / name}.png")


def savetab(df, name):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    print(f"  table  -> {TAB / name}.csv\n{df.round(3).to_string(index=False)}\n")


def knee_sigma(out) -> np.ndarray:
    """Std of the knee angle from the full P: var(th - sh) = P_tt + P_ss - 2 P_ts."""
    P, i, j = out["P"], sk.ANG_THIGH_L, sk.ANG_SHANK_L
    return np.sqrt(P[:, i, i] + P[:, j, j] - 2 * P[:, i, j])


def run_all_methods(kp, conf, p=EKFParams(), ma_window=5, kf_sigma_a=2.0):
    """Keypoints (T, 12, 2) of every method on one sequence, plus the EKF output dict."""
    out = PoseEKF(p).run(kp, conf)
    return {
        "raw": M.mask_low_conf(kp, conf, p.c_min),
        "ma": moving_average(kp, conf, ma_window, p.c_min),
        "kf": kf_per_keypoint(kp, conf, p, sigma_a=kf_sigma_a)[0],
        "ekf": out["z"].reshape(-1, 12, 2),
    }, out


def angle_scores(kp_est, clean):
    """knee / hip RMSE (deg, after warm-up) and knee jitter of one method vs the truth."""
    k_t, h_t = M.joint_angles(clean, SIDE)
    k_e, h_e = M.joint_angles(kp_est, SIDE)
    v = np.arange(len(clean)) >= WARM
    return {"knee_rmse": M.angle_rmse(k_e, k_t, v), "hip_rmse": M.angle_rmse(h_e, h_t, v),
            "knee_jitter": M.jitter(np.where(v, k_e, np.nan))}


# ----------------------------------------------------------------------------- E1
def e1():
    print("E1: knee / hip angle time series")
    X, kp, conf, clean, _ = scenario("clean", seed=0)
    kps, out = run_all_methods(kp, conf)
    t = np.arange(len(X))
    k_t, h_t = (np.degrees(a) for a in M.joint_angles(clean, SIDE))
    k_r, h_r = (np.degrees(a) for a in M.joint_angles(kps["raw"], SIDE))
    k_e, h_e = (np.degrees(a) for a in M.joint_angles(kps["ekf"], SIDE))
    s = np.degrees(knee_sigma(out))
    sc_r, sc_e = angle_scores(kps["raw"], clean), angle_scores(kps["ekf"], clean)

    fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax = axs[0]
    ax.plot(t, k_r, ".", ms=3, color=C["raw"], label=f"raw keypoints (RMSE {sc_r['knee_rmse']:.1f}°)")
    ax.fill_between(t, k_e - 2 * s, k_e + 2 * s, color=C["ekf"], alpha=0.2, label="EKF ± 2σ")
    ax.plot(t, k_e, color=C["ekf"], lw=1.8, label=f"EKF (RMSE {sc_e['knee_rmse']:.1f}°)")
    ax.plot(t, k_t, color=C["truth"], lw=1.2, ls="--", label="truth")
    ax.set_ylabel("knee flexion (deg)"); ax.legend(fontsize=8, ncol=2, frameon=False)
    ax = axs[1]
    ax.plot(t, h_r, ".", ms=3, color=C["raw"], label=f"raw (RMSE {sc_r['hip_rmse']:.1f}°)")
    ax.plot(t, h_e, color=C["ekf"], lw=1.8, label=f"EKF (RMSE {sc_e['hip_rmse']:.1f}°)")
    ax.plot(t, h_t, color=C["truth"], lw=1.2, ls="--", label="truth")
    ax.set_ylabel("hip flexion (deg)"); ax.set_xlabel("frame"); ax.legend(fontsize=8, ncol=3, frameon=False)
    fig.suptitle("E1 (simulation): EKF vs raw keypoints, synthetic squat", fontsize=11)
    inside = np.mean(np.abs(k_e - k_t)[WARM:] <= 2 * s[WARM:])
    print(f"  knee RMSE raw {sc_r['knee_rmse']:.2f} -> EKF {sc_e['knee_rmse']:.2f} deg, "
          f"truth inside ±2σ in {100 * inside:.0f}% of frames")
    savefig(fig, "e1_sim_angles")


# ----------------------------------------------------------------------------- E2
SWEEP_ALPHA, SWEEP_S0, SEEDS = [0.01, 0.03, 0.1], [2.0, 4.0, 8.0], range(10)


def e2():
    print("E2: sigma_alpha x sigma0 sweep (10 seeds each)")
    data = [scenario("clean", seed=s)[1:4] for s in SEEDS]           # (kp, conf, clean)
    rows = []
    for sa in SWEEP_ALPHA:
        for s0 in SWEEP_S0:
            p = EKFParams(sigma_alpha=sa, sigma0=s0)
            sc = [angle_scores(PoseEKF(p).run(kp, conf)["z"].reshape(-1, 12, 2), clean)
                  for kp, conf, clean in data]
            rows.append({"sigma_alpha": sa, "sigma0": s0,
                         **{k: np.mean([d[k] for d in sc]) for k in sc[0]}})
    df = pd.DataFrame(rows)
    savetab(df, "e2_sim_sweep")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title in [(axs[0], "knee_rmse", "knee RMSE (deg)"), (axs[1], "knee_jitter", "knee jitter (deg/frame²)")]:
        G = df.pivot(index="sigma_alpha", columns="sigma0", values=key).values
        im = ax.imshow(G, cmap="viridis_r", origin="lower")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        ax.set_xticks(range(3), [str(v) for v in SWEEP_S0]); ax.set_yticks(range(3), [str(v) for v in SWEEP_ALPHA])
        ax.set_xlabel("σ0 (px)  → trust YOLO less"); ax.set_ylabel("σ_α (rad/frame²)  → allow faster motion")
        ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("E2 (simulation): process vs measurement noise trade-off", fontsize=11)
    savefig(fig, "e2_sim_sweep")


# ----------------------------------------------------------------------------- E3
N_RUNS, T3 = 30, 100
P_MATCHED = EKFParams(sigma_a=0.5, sigma_alpha=0.01, sigma_L=0.02, sigma0=3.0,
                      adaptive_R=False, gating=False, joint_limits=False, reinit_after=0)


def _nees_nis(X, out):
    err = out["x"] - X
    err[:, sk.ANGLES] = sk.wrap(err[:, sk.ANGLES])
    ok = np.isfinite(err).all(1)
    nees = np.full(len(X), np.nan)
    nees[ok] = M.nees(err[ok], out["P"][ok])
    nis = np.where(out["status"] == "update", out["nis"] / np.maximum(out["dof"], 1), np.nan)
    return nees / sk.N_X, nis


def e3():
    print(f"E3: NEES / NIS consistency, {N_RUNS} Monte-Carlo runs")
    res = {}
    # (a) truth drawn from the filter's own model, R matched exactly
    nees, nis = [], []
    for r in range(N_RUNS):
        rng = np.random.default_rng(100 + r)
        x0 = squat_states(10, 45, rng)[0]
        x0[sk.VEL_P0] = 0; x0[sk.VEL_ANG] = 0
        X = cv_states(x0, P_MATCHED, T3, rng)
        kp, conf, _ = observe(X, SimConfig(sigma0=P_MATCHED.sigma0, alpha=0.0, conf_fixed=1.0), rng)
        a, b = _nees_nis(X, PoseEKF(P_MATCHED).run(kp, conf))
        nees.append(a); nis.append(b)
    res["matched model"] = (np.array(nees), np.array(nis))
    # (b) realistic squat, default parameters (model mismatch)
    nees, nis = [], []
    for r in range(N_RUNS):
        X, kp, conf, _, _ = scenario("clean", seed=200 + r, T=T3)
        a, b = _nees_nis(X, PoseEKF(EKFParams()).run(kp, conf))
        nees.append(a); nis.append(b)
    res["squat, default Q/R"] = (np.array(nees), np.array(nis))

    band_x = np.array(M.chi2_band(sk.N_X, N_RUNS)) / sk.N_X
    band_z = np.array(M.chi2_band(2 * sk.N_KP, N_RUNS)) / (2 * sk.N_KP)
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    t = np.arange(T3)
    for (name, (ne, ni)), col, ls in zip(res.items(), [C["ekf"], C["kf"]], ["-", "--"]):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # frame 0 (init) is all NaN
            mne, mni = np.nanmean(ne, 0), np.nanmean(ni, 0)
        axs[0].plot(t, mne, color=col, ls=ls, label=name)
        axs[1].plot(t, mni, color=col, ls=ls, label=name)
        w = t >= WARM
        rows.append({"scenario": name, "mean NEES/dim": np.nanmean(mne[w]), "mean NIS/dof": np.nanmean(mni[w]),
                     "NEES in 95% band (%)": 100 * np.mean((mne[w] >= band_x[0]) & (mne[w] <= band_x[1])),
                     "NIS in 95% band (%)": 100 * np.mean((mni[w] >= band_z[0]) & (mni[w] <= band_z[1]))})
    for ax, band, lab in [(axs[0], band_x, "NEES / 29"), (axs[1], band_z, "NIS / 24")]:
        ax.axhspan(*band, color="#16a34a", alpha=0.15, label="χ² 95 % band")
        ax.axhline(1, color="k", lw=0.8); ax.set_yscale("log"); ax.set_xlabel("frame")
        ax.set_ylabel(f"average {lab} ({N_RUNS} runs)"); ax.legend(fontsize=8, frameon=False)
    axs[0].set_title("state consistency (NEES)", fontsize=10); axs[1].set_title("innovation consistency (NIS)", fontsize=10)
    fig.suptitle("E3 (simulation): is the filter's uncertainty honest?", fontsize=11)
    savefig(fig, "e3_consistency")
    savetab(pd.DataFrame(rows), "e3_consistency")


# ----------------------------------------------------------------------------- E4
OCC_START, OCC_LENS = 60, [10, 20, 30]


def e4():
    print("E4: occlusion of knees + ankles (EKF with / without track-loss recovery vs per-keypoint KF)")
    fig, axs = plt.subplots(2, len(OCC_LENS), figsize=(12, 5.5), sharex=True, sharey="row")
    rows = []
    for c, L in enumerate(OCC_LENS):
        acc = {"EKF": [], "EKF no re-init": [], "per-keypoint KF": [], "sig": [], "sig_nr": []}
        for s in SEEDS:
            X, kp, conf, clean, _ = scenario("occlusion", seed=s, occlusion=[(OCC_START, L, LEG_KP)])
            k_t = M.joint_angles(clean, SIDE)[0]
            err = lambda k: np.degrees(np.abs(M.joint_angles(k, SIDE)[0] - k_t))
            out = PoseEKF(EKFParams()).run(kp, conf)
            out_nr = PoseEKF(EKFParams(reinit_after=0)).run(kp, conf)
            acc["EKF"].append(err(out["z"].reshape(-1, 12, 2)))
            acc["EKF no re-init"].append(err(out_nr["z"].reshape(-1, 12, 2)))
            acc["per-keypoint KF"].append(err(kf_per_keypoint(kp, conf, EKFParams())[0]))
            acc["sig"].append(np.degrees(knee_sigma(out))); acc["sig_nr"].append(np.degrees(knee_sigma(out_nr)))
        m = {k: np.mean(v, 0) for k, v in acc.items()}
        t = np.arange(len(m["EKF"]))
        win, after = slice(OCC_START, OCC_START + L), slice(OCC_START + L, OCC_START + L + 20)
        for name, col, ls in [("EKF", C["ekf"], "-"), ("EKF no re-init", C["ekf"], "--"), ("per-keypoint KF", C["kf"], ":")]:
            axs[0, c].plot(t, m[name], color=col, ls=ls, label=name)
        axs[1, c].plot(t, 2 * m["sig"], color=C["ekf"], label="EKF")
        axs[1, c].plot(t, 2 * m["sig_nr"], color=C["ekf"], ls="--", label="EKF no re-init")
        for ax in axs[:, c]:
            ax.axvspan(OCC_START, OCC_START + L, color="#f97316", alpha=0.1)
        axs[0, c].set_title(f"{L}-frame gap", fontsize=10); axs[1, c].set_xlabel("frame")
        rows.append({"gap (frames)": L,
                     **{f"{n}: mean err in gap": m[n][win].mean() for n in ("EKF", "per-keypoint KF")},
                     **{f"{n}: mean err 20 fr after": m[n][after].mean() for n in ("EKF", "EKF no re-init", "per-keypoint KF")},
                     "EKF 2σ before gap": 2 * m["sig"][OCC_START - 1], "EKF 2σ at gap end": 2 * m["sig"][OCC_START + L - 1]})
    axs[0, 0].set_ylabel("|knee error| (deg), mean of 10 runs"); axs[1, 0].set_ylabel("EKF 2σ knee (deg)")
    axs[0, 0].legend(fontsize=8, frameon=False); axs[1, 0].legend(fontsize=8, frameon=False)
    axs[1, 0].set_yscale("log")
    fig.suptitle("E4 (simulation): knees and ankles hidden from frame 60 (shaded)", fontsize=11)
    savefig(fig, "e4_occlusion")
    savetab(pd.DataFrame(rows), "e4_occlusion")


# ----------------------------------------------------------------------------- E5 (sim)
VARIANTS = {
    "EKF (full)": {},
    "EKF fixed R": {"adaptive_R": False},
    "EKF no gating": {"gating": False},
    "EKF no joint limits": {"joint_limits": False},
    "EKF no re-init": {"reinit_after": 0},
    "EKF fixed R + no gating": {"adaptive_R": False, "gating": False},
}


def e5_sim():
    print("E5 (sim): ablation on the 'realistic' scenario (20 seeds)")
    acc = {}
    for s in range(20):
        X, kp, conf, clean, _ = scenario("realistic", seed=s)
        kps, _ = run_all_methods(kp, conf)
        for name in ("raw", "ma", "kf"):
            acc.setdefault({"raw": "raw", "ma": "moving average", "kf": "per-keypoint KF"}[name], []).append(
                angle_scores(kps[name], clean))
        for name, ov in VARIANTS.items():
            p = dataclasses.replace(EKFParams(), **ov)
            acc.setdefault(name, []).append(angle_scores(PoseEKF(p).run(kp, conf)["z"].reshape(-1, 12, 2), clean))
    df = pd.DataFrame([{"method": k, **{m: np.nanmean([d[m] for d in v]) for m in v[0]}} for k, v in acc.items()])
    savetab(df, "e5_sim_ablation")


EXPS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5_sim}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(EXPS))
    for name in ap.parse_args().only:
        EXPS[name.upper()]()
