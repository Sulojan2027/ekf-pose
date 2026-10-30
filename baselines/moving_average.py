"""
baselines/moving_average.py

Baseline 1: CAUSAL moving average of each keypoint over the last `window` frames.
Causal = frame t may only use frames <= t (the EKF is causal too, so this is fair).
Only keypoints with conf >= c_min are averaged (same threshold as every other method).

Expected behaviour (for the report): less jitter than raw, but a LAG of about
(window - 1) / 2 frames, and no notion of bones (lengths still fluctuate).
"""

import numpy as np


def moving_average(kp: np.ndarray, conf: np.ndarray, window: int = 5,
                   c_min: float = 0.3) -> np.ndarray:
    """Causal masked moving average.

    Args:
        kp:     (T, 12, 2) keypoints (px).
        conf:   (T, 12) confidences.
        window: number of frames averaged (window = 1 reproduces the masked raw input).
        c_min:  keypoints below this confidence are ignored.
    Returns:
        out: (T, 12, 2). out[t, j] = mean of kp[s, j] over s in [t - window + 1, t]
             with conf[s, j] >= c_min;  NaN if there is no such s.
    """
    T = len(kp)
    valid = conf >= c_min
    out = np.full(kp.shape, np.nan)
    for t in range(T):
        lo = max(0, t - window + 1)
        w = valid[lo:t+1]
        n = w.sum(0)
        s = np.where(w[..., None], kp[lo:t+1], 0.0).sum(0)
        out[t, n>0] = s[n>0] / n[n>0, None]
    return out
