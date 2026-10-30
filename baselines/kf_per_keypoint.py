"""
baselines/kf_per_keypoint.py

Baseline 2: an independent LINEAR constant-velocity Kalman filter for every keypoint.
State of keypoint j: s = [x, y, vx, vy] (px, px/frame). Measurement: [x, y].

This is "what you get without a body model": it smooths each point, but knows nothing
about bones, so lengths can still drift and joints can still bend impossibly. It is the
key comparison for the "why an EKF" argument (HANDOFF section 4).

Fairness: same confidence threshold c_min and the same confidence-adaptive sigma_j
(models.measurement_std) as the EKF. No gating (kept deliberately simple).
"""

import numpy as np

from ekf.models import EKFParams, measurement_std

H4 = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])         # measure position only


def make_F4(dt: float = 1.0) -> np.ndarray:
    """(4, 4) constant-velocity transition for [x, y, vx, vy].   [filled]"""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def make_Q4(sigma_a: float, dt: float = 1.0) -> np.ndarray:
    """(4, 4) white-noise-acceleration Q, same derivation as models.make_Q.   [filled]"""
    I2 = np.eye(2)
    return sigma_a ** 2 * np.block([[dt ** 4 / 4 * I2, dt ** 3 / 2 * I2],
                                    [dt ** 3 / 2 * I2, dt ** 2 * I2]])


def kf_per_keypoint(kp: np.ndarray, conf: np.ndarray, p: EKFParams | None = None,
                    sigma_a: float = 2.0, p0_vel_std: float = 10.0):
    """Run 12 independent CV Kalman filters.

    Args:
        kp:      (T, 12, 2) keypoints.  conf: (T, 12).
        p:       EKFParams: uses c_min, dt, sigma0, alpha, adaptive_R (for R).
        sigma_a: keypoint acceleration noise (px/frame^2). Its own knob, tuned on the
                 tuning clips like everything else (keypoints move faster than the root).
        p0_vel_std: initial velocity std (px/frame).
    Returns:
        out: (T, 12, 2) filtered positions, NaN before a keypoint's first valid detection
        var: (T, 12)    position variance P[0, 0] (px^2), NaN before init
    """
    p = p or EKFParams()
    T = len(kp)
    F, Q = make_F4(p.dt), make_Q4(sigma_a, p.dt)
    sigma = measurement_std(conf.reshape(-1), p).reshape(conf.shape)     # (T, 12)
    valid = conf >= p.c_min
    out = np.full((T, 12, 2), np.nan)
    var = np.full((T, 12), np.nan)

    for j in range(12):
        s, P = None, None                       # this keypoint's state (4,) and cov (4, 4)
        for t in range(T):
            if s is not None:                   # predict (only once initialised)
                s = F @ s
                P = F @ P @ F.T + Q

            if valid[t, j]:
                if s is None:                   # first detection: initialise, no update
                    s = np.array([kp[t, j, 0], kp[t, j, 1], 0.0, 0.0])
                    sg = sigma[t, j]            # measurement std (px), not sigma_a
                    P = np.diag([sg**2, sg**2, p0_vel_std**2, p0_vel_std**2])
                else:
                    R = sigma[t, j]**2 * np.eye(2)
                    y = kp[t, j] - H4 @ s
                    S = H4 @ P @ H4.T + R
                    K = np.linalg.solve(S, H4 @ P).T
                    s += K @ y
                    A = np.eye(4) - K @ H4
                    P = A @ P @ A.T + K @ R @ K.T
            if s is not None:                   # store (coasting frames too)
                out[t, j] = s[:2]
                var[t, j] = P[0, 0]
    return out, var
