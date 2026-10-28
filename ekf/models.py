"""
ekf/models.py

Process and measurement noise models for the pose EKF (HANDOFF 7.5, 7.6):
  - EKFParams:           all tunable parameters in one place (+ ablation switches for E5)
  - make_F():            constant-velocity transition matrix (29 x 29)
  - make_Q():            discrete white-noise-acceleration process noise (29 x 29)
  - measurement_std():   confidence-adaptive keypoint noise sigma_j
  - kept_rows(), make_R(): measurement rows / noise for the keypoints kept this frame

Units: dt = 1 frame, so sigma_a is px/frame^2, sigma_alpha is rad/frame^2.
"""

from dataclasses import dataclass

import numpy as np

from ekf import skeleton as sk
N_X = 29

@dataclass
class EKFParams:
    """Tunable parameters. Starting values from HANDOFF 7.5; tune on the tuning clips only."""
    sigma_a: float = 1.0        # root acceleration noise, px/frame^2
    sigma_alpha: float = 0.03   # angular acceleration noise, rad/frame^2
    sigma_L: float = 0.05       # length random walk, px/frame
    sigma0: float = 4.0         # keypoint noise at confidence 1, px
    alpha: float = 4.0          # how fast noise grows as confidence drops
    c_min: float = 0.3          # keypoints below this confidence are not used
    gate: float = 9.21          # chi2(2 dof) 99% quantile for per-keypoint NIS gating
    dt: float = 1.0
    # ablation switches (E5)
    adaptive_R: bool = True     # False -> sigma_j = sigma0 for every kept keypoint
    gating: bool = True
    joint_limits: bool = True


def make_F(dt: float = 1.0) -> np.ndarray:
    """Constant-velocity transition matrix.

    Returns:
        F: (29, 29). Identity, plus dt coupling every position/angle i (0..10)
           to its velocity i + VEL_OFFSET:   x_i <- x_i + dt * x_{i+11}.
           Velocities and lengths map to themselves.
    """
    F = np.eye(sk.N_X)
    
    for i in range(sk.VEL_OFFSET):
        F[i, i + sk.VEL_OFFSET] = dt
    return F


def make_Q(p: EKFParams) -> np.ndarray:
    """Discrete white-noise-acceleration process noise.

    For each position/angle i (0..10) and its velocity v = i + VEL_OFFSET:
        Q[[i, v]][:, [i, v]] = sigma^2 * [[dt^4/4, dt^3/2],
                                         [dt^3/2, dt^2  ]]
        sigma = p.sigma_a for the 2 root coords (i = 0, 1),
        sigma = p.sigma_alpha for the 9 angles (i = 2..10).
    Lengths (22..28): Q[l, l] = p.sigma_L^2.
    Returns:
        Q: (29, 29), symmetric PSD.
    """
    dt = p.dt
    Q = np.eye(sk.N_X)
    G = [[dt**4/4, dt**3/2], [dt**3/2, dt**2]]
    for i in range(sk.VEL_OFFSET):
        idx = [i, i + sk.VEL_OFFSET]
        sigma = p.sigma_a if i in (0, 1) else p.sigma_alpha
        Q[np.ix_(idx, idx)] = (sigma ** 2 )* G
    for l in range(sk.LEN_TORSO, sk.N_X):
        Q[l, l] = p.sigma_L ** 2
        
    return Q

def measurement_std(conf: np.ndarray, p: EKFParams) -> np.ndarray:
    """Per-keypoint measurement std (px).

    Args:
        conf: (12,) YOLO confidences in [0, 1].
    Returns:
        sigma: (12,)   sigma_j = sigma0 * sqrt(1 + alpha * (1 - c_j))   if p.adaptive_R
                       sigma_j = sigma0                                 otherwise
    """
    sigma = np.zeros_like(conf)
    for idx, c in enumerate(conf):
        sigma[idx] = p.sigma0 * np.sqrt(1 + p.alpha * (1 - c)) if p.adaptive_R else p.sigma0
    return sigma

def kept_rows(mask: np.ndarray) -> np.ndarray:
    """Measurement row indices of the kept keypoints.   [glue]

    mask (12,) bool -> rows (2 * n_kept,): keypoint j contributes rows 2j, 2j+1.
    Example: mask True at j = 8, 10 -> [16, 17, 20, 21].
    """
    return np.flatnonzero(np.repeat(mask, 2))


def make_R(sigma: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Measurement noise for the kept keypoints only.

    Args:
        sigma: (12,) per-keypoint std from measurement_std().
        mask:  (12,) bool, keypoints kept this frame.
    Returns:
        R: (2k, 2k) diagonal, k = mask.sum(); x and y of keypoint j both get sigma_j^2,
           in the same row order as kept_rows(mask).
    """
    return np.diag(np.repeat(sigma[mask] ** 2, 2))
   
