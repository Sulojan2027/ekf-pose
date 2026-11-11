"""
eval/metrics.py
"""

import numpy as np
from scipy.stats import chi2

from ekf import skeleton as sk

DEG = 180.0 / np.pi
SIDE_KP = {  # (shoulder, hip, knee, ankle) in local keypoint order
    "L": (sk.L_SH, sk.L_HIP, sk.L_KN, sk.L_AN),
    "R": (sk.R_SH, sk.R_HIP, sk.R_KN, sk.R_AN),
}


# ----------------------------------------------------------------------------- angles
def joint_angles(kp: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Knee and hip flexion (rad) of one leg from keypoints, for ANY method.
    """
    sh, hp, kn, an = (kp[:, j] for j in SIDE_KP[side])
    return sk.flexion_angles(sk.seg_angle(hp, sh), sk.seg_angle(hp, kn), sk.seg_angle(kn, an))


def angle_valid(vis: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """GT masks (T,) bool: knee needs hip+knee+ankle visible, hip needs shoulder+hip+knee."""
    sh, hp, kn, an = SIDE_KP[side]
    return vis[:, [hp, kn, an]].all(1), vis[:, [sh, hp, kn]].all(1)


def mask_low_conf(kp: np.ndarray, conf: np.ndarray, c_min: float) -> np.ndarray:
    """Raw method: copy of kp with NaN where conf < c_min (YOLO zeros are not positions)."""
    out = kp.astype(float).copy()
    out[conf < c_min] = np.nan
    return out


# ----------------------------------------------------------------------------- scores
def angle_rmse(est: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """RMSE (deg) between two angle series on frames where valid AND both are finite.
    """
    m = valid & np.isfinite(est) & np.isfinite(gt)
    if not m.any():
        return np.nan
    err = sk.wrap(est[m] - gt[m])              
    return float(np.sqrt(np.mean(err ** 2)) * DEG)


def jitter(angle: np.ndarray) -> float:
    """Mean absolute second difference (deg / frame^2): how 'shaky' a curve is.
    """
    d2 = angle[2:] - 2 * angle[1:-1] + angle[:-2]
    d2 = d2[np.isfinite(d2)]
    if d2.size == 0:
        return np.nan
    return float(np.mean(np.abs(d2)) * DEG)


def lag(est: np.ndarray, gt: np.ndarray, valid: np.ndarray, max_lag: int = 10) -> int | float:
    """Delay of est behind gt (frames) = the shift k that maximises their cross-correlation.
    """
    ok = valid & np.isfinite(est) & np.isfinite(gt)
    T = len(est)
    if ok.sum() < 10:
        return np.nan
    a = np.where(ok, est - est[ok].mean(), np.nan)
    b = np.where(ok, gt - gt[ok].mean(), np.nan)
    score = {}
    for k in range(-max_lag, max_lag + 1):
        aa, bb = (a[k:], b[:T-k]) if k >= 0 else (a[:T+k], b[-k:])
        m = np.isfinite(aa) & np.isfinite(bb)
        if m.sum() < 10:
            continue
        score[k] = np.mean(aa[m] * bb[m])
    if not score:
        return np.nan
    return int(max(score, key=score.get))


def keypoint_rmse_pct(kp_est: np.ndarray, kp_gt: np.ndarray, vis: np.ndarray,
                      bbox: np.ndarray) -> float:
    """Keypoint RMSE as % of the GT person-box height (scale-free across clips).
    """
    h = bbox[:, 3] - bbox[:, 1]
    e = np.linalg.norm(kp_est - kp_gt, axis=-1) / h[:, None]
    m = vis & np.isfinite(e)
    if not m.any():
        return np.nan
    return float(100 * np.sqrt(np.mean(e[m] ** 2)))


def bone_length_cv(kp: np.ndarray, side: str, valid: np.ndarray | None = None) -> float:
    """How much the thigh and shank lengths implied by kp vary: mean of the two CVs (%).
    """
    _, hip, knee, an = SIDE_KP[side]
    cvs = []                                      
    for a, b in ((hip, knee), (knee, an)):
        L = np.linalg.norm(kp[:, b] - kp[:, a], axis=-1)      
        m = np.isfinite(L) if valid is None else (np.isfinite(L) & valid)
        cvs.append(L[m].std() / L[m].mean() if m.sum() >= 2 else np.nan)
    return float(100 * np.mean(cvs))              


def nees(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Normalised Estimation Error Squared per frame: e^T P^-1 e.
    """
    sol = np.linalg.solve(P, err[..., None])[..., 0]
    return (err * sol).sum(-1)


def chi2_band(dof: int, n_runs: int, q: float = 0.95) -> tuple[float, float]:
    """Two-sided q-probability band for the AVERAGE of n_runs independent chi^2(dof) values.
    """
    n = n_runs * dof
    lo = chi2.ppf((1 - q) / 2, n) / n_runs
    hi = chi2.ppf((1 + q) / 2, n) / n_runs
    return float(lo), float(hi)
