"""
sim/simulate.py

Synthetic side-view squats with KNOWN ground truth, for the simulation experiments
(E1-E5, experiments/exp_sim.py). Nothing here touches YOLO or Penn Action.

  - SimConfig:       every knob of the synthetic "YOLO"                       [filled]
  - squat_states():  smooth, realistic squat motion (truth for E1, E2, E4, E5) [filled]
  - cv_states():     truth drawn FROM the filter's own process model (E3)     [done]     
  - observe():       truth -> noisy keypoints + confidences, then dropouts,
                     occlusion windows and confident outliers               [done]
  - scenario():      named, reproducible scenarios used by exp_sim.py         [filled]

Synthetic detector model (what we PRETEND YOLO does):
    c_j          ~ Beta(conf_a, conf_b)                      per keypoint and frame
    sigma_true_j = sigma0 * sqrt(1 + alpha * (1 - c_j))      same FORM as the filter's R,
    kp_j         = fk(x)_j + N(0, sigma_true_j^2 I_2)        but its own sigma0 / alpha
Report note: because the simulated noise follows the same form as the adaptive R, the
simulation is a best case for adaptive R; the real-data ablation (E5) is the honest test.
"""

from dataclasses import dataclass, field

import numpy as np

from ekf import skeleton as sk
from ekf.models import EKFParams, make_F, make_Q

LEG_KP = (sk.L_KN, sk.R_KN, sk.L_AN, sk.R_AN)       # keypoints hidden in the occlusion test


@dataclass
class SimConfig:
    """Synthetic detector + motion settings."""
    T: int = 150                 # frames
    period: float = 45.0         # frames per squat repetition
    sigma0: float = 3.0          # TRUE keypoint noise at confidence 1 (px)
    alpha: float = 4.0           # TRUE noise growth as confidence drops
    conf_a: float = 8.0          # confidences ~ Beta(conf_a, conf_b): mean = a / (a + b)
    conf_b: float = 1.5
    conf_fixed: float | None = None   # if set, every confidence equals this (E3)
    p_drop: float = 0.0          # chance that a keypoint is "not detected" (conf -> 0)
    occlusion: list = field(default_factory=list)   # [(start, length, keypoints), ...] -> conf 0
    p_outlier: float = 0.0       # chance per frame of ONE confident but wrong leg keypoint
    outlier_px: float = 40.0     # size of that error (random direction)
    seed: int = 0


# ----------------------------------------------------------------------------- truth
def squat_states(T: int, period: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth side-view squat. Returns X (T, 29) with consistent velocities.   [filled]

    Random per call: body size, squat depth, facing direction (left / right).
    The near-side ankle is kept fixed on the floor and the root p0 is solved from it,
    so the hips drop and move back exactly as the leg angles dictate.
    """
    t = np.arange(T)
    depth = rng.uniform(0.7, 1.0)
    s = depth * (1 - np.cos(2 * np.pi * t / period)) / 2            # 0 standing -> depth
    wob = 0.03 * np.sin(2 * np.pi * t / (period * 0.37) + rng.uniform(0, 6))   # small natural wobble

    L_t = rng.uniform(70, 90)
    lengths = np.array([L_t, 3.0, 3.0, 0.62 * L_t, 0.49 * L_t, 0.82 * L_t, 0.82 * L_t])
    lengths[3:] *= rng.uniform(0.95, 1.05, 4)

    X = np.zeros((T, sk.N_X))
    X[:, sk.ANG_TORSO] = -np.pi / 2 + 0.6 * s + wob                  # leans forward (+x)
    for th, sh in ((sk.ANG_THIGH_L, sk.ANG_SHANK_L), (sk.ANG_THIGH_R, sk.ANG_SHANK_R)):
        X[:, th] = np.pi / 2 - 1.3 * s                               # knee forward
        X[:, sh] = np.pi / 2 + 0.6 * s                               # shank tilts back down
    for ua, fa in ((sk.ANG_UARM_L, sk.ANG_FARM_L), (sk.ANG_UARM_R, sk.ANG_FARM_R)):
        X[:, ua] = np.pi / 2 - 1.4 * s + wob                         # arms reach forward
        X[:, fa] = np.pi / 2 - 1.5 * s + wob
    if rng.random() < 0.5:                                           # face left: mirror x
        X[:, sk.ANGLES] = np.pi - X[:, sk.ANGLES]
    X[:, sk.ANGLES] = sk.wrap(X[:, sk.ANGLES])
    X[:, sk.LENGTHS] = lengths

    ankle = np.array([rng.uniform(180, 260), rng.uniform(380, 420)])
    for k in range(T):                                               # p0 = ankle - leg chain
        x = X[k].copy(); x[sk.P0] = 0.0
        X[k, sk.P0] = ankle - sk.fk(x).reshape(sk.N_KP, 2)[sk.L_AN]
    X[:, sk.VEL_P0] = np.gradient(X[:, sk.P0], axis=0)
    X[:, sk.VEL_ANG] = np.gradient(np.unwrap(X[:, sk.ANGLES], axis=0), axis=0)
    return X


def cv_states(x0: np.ndarray, p: EKFParams, T: int, rng: np.random.Generator) -> np.ndarray:
    """Truth that follows the filter's process model EXACTLY (for the E3 consistency test).

    x_{k+1} = F x_k + w_k,   w_k ~ N(0, Q)          (F = make_F(p.dt), Q = make_Q(p))

    Args:
        x0: (29,) start state (e.g. frame 0 of squat_states with velocities zeroed).
        p:  the SAME EKFParams the filter will use (that is the point of E3).
    Returns:
        X: (T, 29), X[0] = x0.
    Why: NEES / NIS can only be expected to match chi^2 when the truth really is
    generated by the model the filter assumes. On squat_states it will not (model mismatch).
    """
    F, Q = make_F(p.dt), make_Q(p)
    X = np.zeros((T, sk.N_X))
    X[0] = x0
    for k in range(1, T):
        w = rng.multivariate_normal(np.zeros(sk.N_X), Q)
        X[k] = sk.wrap_angles(F @ X[k-1] + w)
    return X


# ----------------------------------------------------------------------------- detector
def observe(X: np.ndarray, cfg: SimConfig, rng: np.random.Generator):
    """Turn true states into what a keypoint detector would output.

    Args:
        X:   (T, 29) true states.
        cfg: SimConfig.
    Returns:
        kp:    (T, 12, 2) noisy keypoints (px)
        conf:  (T, 12)    confidences in [0, 1] (0 = not detected)
        clean: (T, 12, 2) noise-free keypoints fk(x) (for keypoint-error metrics)
    Order: confidences -> noise from confidences -> dropouts / occlusion (conf 0) -> outliers.
    """
    T = len(X)
    clean = np.stack([sk.fk(x).reshape(sk.N_KP, 2) for x in X])
    if cfg.conf_fixed is not None:
        conf = np.full((T, sk.N_KP), float(cfg.conf_fixed))
    else:
        conf = rng.beta(cfg.conf_a, cfg.conf_b, size=(T, sk.N_KP))
    
    sigma = cfg.sigma0 * np.sqrt(1 + cfg.alpha * (1 - conf))
    kp = clean + sigma[..., None] * rng.standard_normal((T, sk.N_KP, 2))

    # random missed keypoints and occlusion windows: the detector reports nothing
    conf[rng.random((T, sk.N_KP)) < cfg.p_drop] = 0.0
    for start, length, kps in cfg.occlusion:
        conf[start:start + length, list(kps)] = 0.0

    # confident outliers: one leg keypoint jumps outlier_px in a random direction
    for t in np.flatnonzero(rng.random(T) < cfg.p_outlier):
        j = rng.choice(LEG_KP)
        ang = rng.uniform(0, 2 * np.pi)
        kp[t, j] += cfg.outlier_px * np.array([np.cos(ang), np.sin(ang)])
        conf[t, j] = max(conf[t, j], 0.8)
    return kp, conf, clean


# ----------------------------------------------------------------------------- scenarios
def scenario(name: str, seed: int = 0, **overrides):
    """Named, reproducible scenarios.   [filled]

    clean      : moderate noise, no misses                        (E1, E2)
    occlusion  : knees + ankles hidden for 20 frames from frame 60  (E4; override occlusion=...)
    realistic  : lower confidences, 5% missed keypoints, 4% confident outliers   (E5)
    Returns (X, kp, conf, clean, cfg).
    """
    base = dict(seed=seed)
    if name == "occlusion":
        base["occlusion"] = [(60, 20, LEG_KP)]
    elif name == "realistic":
        base.update(conf_a=4.0, conf_b=1.5, p_drop=0.05, p_outlier=0.04)
    elif name != "clean":
        raise ValueError(name)
    base.update(overrides)
    cfg = SimConfig(**base)
    rng = np.random.default_rng(cfg.seed)
    X = squat_states(cfg.T, cfg.period, rng)
    kp, conf, clean = observe(X, cfg, rng)
    return X, kp, conf, clean, cfg
