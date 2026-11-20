"""
Process and measurement noise models for the pose EKF,
Units: dt = 1 frame, so sigma_a is px/frame^2, sigma_alpha is rad/frame^2.
"""

from dataclasses import dataclass

import numpy as np

from ekf import skeleton as sk
N_X = 29

@dataclass
class EKFParams:
    """Tunable parameters. Tuned on the tuning clips only."""
    sigma_a: float = 1.0        # root acceleration noise, px/frame^2
    sigma_alpha: float = 0.03   # angular acceleration noise, rad/frame^2
    sigma_L: float = 0.05       # length random walk, px/frame
    sigma0: float = 4.0         # keypoint noise at confidence 1, px
    alpha: float = 4.0          # how fast noise grows as confidence drops
    c_min: float = 0.3          # keypoints below this confidence are not used
    gate: float = 9.21          # chi2(2 dof)
    dt: float = 1.0
    adaptive_R: bool = True 
    gating: bool = True
    joint_limits: bool = True
    reinit_after: int = 3


def make_F(dt: float = 1.0) -> np.ndarray:
    """Constant-velocity transition matrix.

    Returns:
        F: (29, 29). Identity, plus dt coupling every position/angle i (0..10)
    """
    F = np.eye(sk.N_X)
    
    for i in range(sk.VEL_OFFSET):
        F[i, i + sk.VEL_OFFSET] = dt
    return F


def make_Q(p: EKFParams) -> np.ndarray:
    """Discrete white-noise-acceleration process noise.
    """
    dt = p.dt
    Q = np.zeros((sk.N_X, sk.N_X))
    G = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
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
        sigma: (12,)
    """
    conf = np.clip(np.asarray(conf, dtype=float), 0.0, 1.0)
    sigma = np.zeros(len(conf))
    for idx, c in enumerate(conf):
        sigma[idx] = p.sigma0 * np.sqrt(1 + p.alpha * (1 - c)) if p.adaptive_R else p.sigma0
    return sigma

def kept_rows(mask: np.ndarray) -> np.ndarray:
    """Measurement row indices of the kept keypoints.
    """
    return np.flatnonzero(np.repeat(mask, 2))


def make_R(sigma: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Measurement noise for the kept keypoints only.
    """
    return np.diag(np.repeat(sigma[mask] ** 2, 2))
   
