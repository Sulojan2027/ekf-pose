"""
eval/metrics.py

Every number in the results tables comes from this file, computed IDENTICALLY for
GT, raw YOLO, the baselines and the EKF (HANDOFF 7.9 / section 9).

  [filled]  joint_angles(), angle_valid(), mask_low_conf()   - how angles are obtained
  [done]    angle_rmse(), jitter(), lag(), keypoint_rmse_pct(),
            bone_length_cv(), nees(), chi2_band()            - how they are scored

Conventions
  - Angles in radians inside, DEGREES out of every metric (tables are in degrees).
  - NaN means "no estimate" (method has nothing at that frame); metrics skip NaNs.
  - `valid` masks come from GT visibility: score only where GT is trustworthy.
  - Side views: evaluate the near-side leg `side` ("L"/"R", from results/clips_*.csv).
"""

import numpy as np
from scipy.stats import chi2

from ekf import skeleton as sk

DEG = 180.0 / np.pi
SIDE_KP = {  # (shoulder, hip, knee, ankle) in local keypoint order
    "L": (sk.L_SH, sk.L_HIP, sk.L_KN, sk.L_AN),
    "R": (sk.R_SH, sk.R_HIP, sk.R_KN, sk.R_AN),
}


# ----------------------------------------------------------------------------- angles  [filled]
def joint_angles(kp: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Knee and hip flexion (rad) of one leg from keypoints, for ANY method.

    kp: (T, 12, 2), NaN allowed.  Returns knee (T,), hip (T,) (NaN where an input is NaN).
    Torso direction = same-side hip -> shoulder (works for GT, where the far shoulder is
    often invisible). For the EKF pass out["z"] (= fk(x)) so it is scored the same way.
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


# ----------------------------------------------------------------------------- scores  [done]
def angle_rmse(est: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """RMSE (deg) between two angle series on frames where valid AND both are finite.

    est, gt: (T,) rad.  valid: (T,) bool.  Returns NaN if no frame qualifies.
    """
    m = valid & np.isfinite(est) & np.isfinite(gt)
    if not m.any():
        return np.nan
    err = sk.wrap(est[m] - gt[m])              # plain angles -> wrap (wrap_angles is for 29-D states)
    return float(np.sqrt(np.mean(err ** 2)) * DEG)


def jitter(angle: np.ndarray) -> float:
    """Mean absolute second difference (deg / frame^2): how 'shaky' a curve is.

    a''(t) ~ a[t+1] - 2 a[t] + a[t-1]. A smooth squat has small a''; noise makes it large.
    Uses only triples of consecutive finite values. NaN if none.
    """
    d2 = angle[2:] - 2 * angle[1:-1] + angle[:-2]
    d2 = d2[np.isfinite(d2)]
    if d2.size == 0:
        return np.nan
    return float(np.mean(np.abs(d2)) * DEG)


def lag(est: np.ndarray, gt: np.ndarray, valid: np.ndarray, max_lag: int = 10) -> int | float:
    """Delay of est behind gt (frames) = the shift k that maximises their cross-correlation.

    Positive k means est is LATE: est[t] best matches gt[t - k] (e.g. a moving average).
    Causal filters can only be late (k >= 0); report it next to RMSE, because smoothing
    buys low jitter by adding lag.
    Returns NaN (not 0) if no shift has >= 10 overlapping frames, so such clips are
    skipped by nanmedian instead of being counted as 'no lag'.
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

    kp_est, kp_gt: (T, 12, 2).  vis: (T, 12) GT visibility.  bbox: (T, 4) [x1, y1, x2, y2].
    Per visible joint: e = ||est - gt|| / box_height(t). Return 100 * sqrt(mean(e^2))
    over visible joints with a finite estimate.
    """
    h = bbox[:, 3] - bbox[:, 1]
    e = np.linalg.norm(kp_est - kp_gt, axis=-1) / h[:, None]
    m = vis & np.isfinite(e)
    if not m.any():
        return np.nan
    return float(100 * np.sqrt(np.mean(e[m] ** 2)))


def bone_length_cv(kp: np.ndarray, side: str, valid: np.ndarray | None = None) -> float:
    """How much the thigh and shank lengths implied by kp vary: mean of the two CVs (%).

    CV = std / mean of a bone's length over frames. A rigid body has CV ~ 0 (up to
    perspective). Raw keypoints and per-keypoint KFs do not enforce this; the EKF does.
    valid: (T,) bool frames to use (e.g. GT knee-visible frames), None = all finite frames.
    """
    _, hip, knee, an = SIDE_KP[side]
    cvs = []                                         # one CV per bone (thigh, shank)
    for a, b in ((hip, knee), (knee, an)):
        L = np.linalg.norm(kp[:, b] - kp[:, a], axis=-1)       # (T,) bone length per frame
        m = np.isfinite(L) if valid is None else (np.isfinite(L) & valid)
        cvs.append(L[m].std() / L[m].mean() if m.sum() >= 2 else np.nan)
    return float(100 * np.mean(cvs))                 # NaN if either bone had < 2 frames


def nees(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Normalised Estimation Error Squared per frame: e^T P^-1 e.

    err: (T, n) estimation error x_est - x_true (ANGLE parts already wrapped by the caller).
    P:   (T, n, n) the filter's covariance.  Returns (T,).
    A consistent filter has E[NEES] = n.
    """
    sol = np.linalg.solve(P, err[..., None])[..., 0]
    return (err * sol).sum(-1)


def chi2_band(dof: int, n_runs: int, q: float = 0.95) -> tuple[float, float]:
    """Two-sided q-probability band for the AVERAGE of n_runs independent chi^2(dof) values.

    Sum of n_runs chi^2(dof) values is chi^2(n_runs * dof), so the average lies in
        [chi2.ppf((1 - q)/2, n_runs*dof), chi2.ppf((1 + q)/2, n_runs*dof)] / n_runs
    with probability q. Used for the E3 plots (NEES and NIS averaged over Monte-Carlo runs).
    """
    n = n_runs * dof
    lo = chi2.ppf((1 - q) / 2, n) / n_runs
    hi = chi2.ppf((1 + q) / 2, n) / n_runs
    return float(lo), float(hi)
