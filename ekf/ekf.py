"""
ekf/ekf.py

Extended Kalman Filter on the 29-D planar skeleton (HANDOFF 7.6, guide Step 7).

Per frame:  predict -> (1) confidence mask -> (2) adaptive R -> (3) per-keypoint NIS gate
            -> EKF update on the kept rows (Joseph form) -> constrain (wrap, symmetrise,
            clamp lengths, joint limits).
If no keypoint survives (1)-(3), the update is skipped and the filter coasts.

Track-loss recovery (added after experiment E4): after a long coast the angle uncertainty is
a curved "banana" that the EKF's linearised Gaussian cannot represent. The returning keypoints
then fail the gate every frame and the limb is never re-acquired. Fix: a limb whose confident
keypoints were gated `reinit_after` frames in a row is re-initialised from the measurement.

Usage:
    ekf = PoseEKF(EKFParams())
    out = ekf.run(kp_seq, conf_seq)     # kp_seq (T, 12, 2), conf_seq (T, 12)
"""

import time

import numpy as np

from ekf import skeleton as sk
from ekf.models import EKFParams, kept_rows, make_F, make_Q, make_R, measurement_std

# limb segments for track-loss recovery: (angle index, proximal kp, distal kp)
LIMB_SEGS = [
    (sk.ANG_UARM_L, sk.L_SH, sk.L_EL), (sk.ANG_FARM_L, sk.L_EL, sk.L_WR),
    (sk.ANG_UARM_R, sk.R_SH, sk.R_EL), (sk.ANG_FARM_R, sk.R_EL, sk.R_WR),
    (sk.ANG_THIGH_L, sk.L_HIP, sk.L_KN), (sk.ANG_SHANK_L, sk.L_KN, sk.L_AN),
    (sk.ANG_THIGH_R, sk.R_HIP, sk.R_KN), (sk.ANG_SHANK_R, sk.R_KN, sk.R_AN),
]


class PoseEKF:
    def __init__(self, params: EKFParams | None = None):
        self.p = params or EKFParams()
        self.F = make_F(self.p.dt)
        self.Q = make_Q(self.p)
        self.x = None                     # (29,) state, None until initialised
        self.P = None                     # (29, 29)
        self.n_rej = np.zeros(sk.N_KP, int)   # frames in a row each keypoint was gated

    @property
    def initialised(self) -> bool:
        return self.x is not None

    # ------------------------------------------------------------------ predict
    def predict(self) -> None:
        """x <- F x,  P <- F P F^T + Q,  then wrap the angles."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x = sk.wrap_angles(self.x)
        return None

    # ------------------------------------------------------------------ gating
    def gate(self, kp: np.ndarray, mask: np.ndarray, z_pred: np.ndarray,
             H: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Per-keypoint NIS gate (step 3). Returns a NEW mask with outliers removed.

        For every keypoint j still in `mask`:
            r     = [2j, 2j+1]
            y_j   = kp[j] - z_pred[r]                               (2,)
            S_jj  = H[r] @ P @ H[r].T + sigma_j^2 * I_2             (2, 2)
            nis_j = y_j @ solve(S_jj, y_j)
            drop j if nis_j > self.p.gate
        Note S_jj is the 2x2 diagonal block of the full S: it asks "is THIS keypoint
        plausible on its own?", so one bad keypoint can't hide behind good ones.
        """
        mask = mask.copy()
        for j in np.flatnonzero(mask):
            r = [2 * j, 2 * j + 1]
            y_j = kp[j] - z_pred[r]
            S_jj = (H[r] @ self.P @ H[r].T) + sigma[j]**2 * np.eye(2)
            nis_j = y_j @ np.linalg.solve(S_jj, y_j)
            if nis_j > self.p.gate:
                mask[j] = False
        return mask

    # ------------------------------------------------------------------ track-loss recovery
    def reinit_lost_segments(self, kp: np.ndarray, mask: np.ndarray) -> int:
        """Re-initialise limb angles whose keypoints have been gated too long.

        For every (a, i, k) in LIMB_SEGS (angle index, proximal kp, distal kp):
          if BOTH ends are usable now (mask[i] and mask[k]) and at least one end has been
          gated >= self.p.reinit_after frames in a row (self.n_rej):
            - angle      x[a]      = seg_angle(kp[i], kp[k])    (trust the measurement)
            - velocity   x[a + 11] = 0                          (unknown again)
            - covariance: forget everything about a and a + 11:
                zero their rows AND columns in P, then put the initial variances back
                (sk.P0_STD_ANG ** 2 and sk.P0_STD_VANG ** 2) on the diagonal
            - reset self.n_rej[i] and self.n_rej[k] to 0
        Returns the number of segments re-initialised.
        Report: this is a re-initialisation heuristic; the principled alternatives are a UKF /
        particle filter (can represent the curved uncertainty) or a multiple-hypothesis tracker.
        """
        n = 0
        for a, i, k in LIMB_SEGS:
            usable = mask[i] and mask[k]
            lost = max(self.n_rej[i], self.n_rej[k]) >= self.p.reinit_after
            if not (usable and lost):
                continue
            v = a + sk.VEL_OFFSET
            self.x[a] = sk.seg_angle(kp[i], kp[k])      # trust the measurement
            self.x[v] = 0.0                             # angular velocity unknown again
            for idx in (a, v):                          # forget all correlations first ...
                self.P[idx, :] = 0.0
                self.P[:, idx] = 0.0
            self.P[a, a] = sk.P0_STD_ANG ** 2           # ... THEN restore the variances
            self.P[v, v] = sk.P0_STD_VANG ** 2          #     (other order -> singular P)
            self.n_rej[i] = self.n_rej[k] = 0
            n += 1
        return n

    # ------------------------------------------------------------------ update
    def update(self, kp: np.ndarray, conf: np.ndarray) -> dict:
        """Measurement update with keypoints of one frame.

        Args:
            kp:   (12, 2) keypoints (local order, px).
            conf: (12,) confidences.
        Returns:
            info dict: status ("wait" | "init" | "update" | "coast"), n_used, n_gated,
                       nis (float, NaN if no update), dof (int, 2 * n_used).
        """
        info = {"status": "coast", "n_used": 0, "n_gated": 0, "nis": np.nan, "dof": 0, "n_reinit": 0}

        mask = conf >= self.p.c_min
        
        if not self.initialised:
            res = sk.init_from_keypoints(kp, mask)
            if res is None:
                info["status"] = "wait"
                return info
            self.x, self.P = res
            info["status"] = "init"
            return info
        
        # track-loss recovery (before linearising, so the new angles are used)   [glue]
        if self.p.reinit_after > 0:
            info["n_reinit"] = self.reinit_lost_segments(kp, mask)

        # linearise at the prediction
        z_pred = sk.fk(self.x)
        H = sk.jacobian(self.x)

        # Adaptive R  and NIS gate
        sigma = measurement_std(conf, self.p)
        if self.p.gating:
            new = self.gate(kp, mask, z_pred, H, sigma)
            info["n_gated"] = int(mask.sum() - new.sum())
            self.n_rej[mask & ~new] += 1          # confident but rejected: count   [glue]
            self.n_rej[new] = 0                   # accepted: reset
            mask = new
        if not(mask.any()):
            return info

        # EKF update on the kept rows only
        rows = kept_rows(mask)
        z = kp.reshape(-1)[rows]
        y = z - z_pred[rows]
        Hk = H[rows]
        R = make_R(sigma, mask)
        S = Hk @ self.P @ Hk.T + R
        K = np.linalg.solve(S, Hk @ self.P).T
        self.x = self.x + K @ y
        A = np.eye(sk.N_X) - K @ Hk
        self.P = A @ self.P @ A.T + K @ R @ K.T
        info["nis"] = float(y @ np.linalg.solve(S, y))
        info["dof"] = len(rows)
        info["n_used"] = int(mask.sum())
        info["status"] = "update"
        
        # constrain
        self.x = sk.clamp_lengths(sk.wrap_angles(self.x))
        if self.p.joint_limits:
            self.x = sk.apply_joint_limits(self.x)
        self.P = 0.5 * (self.P + self.P.T)
        return info

    # ------------------------------------------------------------------ driver   [glue]
    def step(self, kp: np.ndarray, conf: np.ndarray) -> dict:
        """One video frame: predict (if initialised) then update."""
        if self.initialised:
            self.predict()
        return self.update(kp, conf)

    def run(self, kp_seq: np.ndarray, conf_seq: np.ndarray) -> dict:
        """Filter a whole clip.   [glue]

        Args:
            kp_seq:   (T, 12, 2) keypoints.
            conf_seq: (T, 12) confidences (0 where nobody was detected).
        Returns dict of arrays (NaN before initialisation):
            x (T, 29), P (T, 29, 29), P_diag (T, 29), z (T, 24) = fk(x),
            nis (T,), dof (T,), n_used (T,), n_gated (T,), n_reinit (T,), status (T,) str, t_ms (T,)
        """
        T = len(kp_seq)
        out = {
            "x": np.full((T, sk.N_X), np.nan), "P": np.full((T, sk.N_X, sk.N_X), np.nan),
            "P_diag": np.full((T, sk.N_X), np.nan),
            "z": np.full((T, sk.N_Z), np.nan), "nis": np.full(T, np.nan),
            "dof": np.zeros(T, int), "n_used": np.zeros(T, int), "n_gated": np.zeros(T, int), "n_reinit": np.zeros(T, int),
            "status": np.empty(T, object), "t_ms": np.zeros(T),
        }
        for t in range(T):
            t0 = time.perf_counter()
            info = self.step(kp_seq[t], conf_seq[t])
            out["t_ms"][t] = (time.perf_counter() - t0) * 1e3
            for k in ("nis", "dof", "n_used", "n_gated", "n_reinit", "status"):
                out[k][t] = info[k]
            if self.initialised:
                out["x"][t] = self.x
                out["P"][t] = self.P
                out["P_diag"][t] = np.diag(self.P)
                out["z"][t] = sk.fk(self.x)
        return out
