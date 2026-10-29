"""
tests/test_ekf.py

Quick sanity checks of the EKF on a synthetic side-view squat (no YOLO, no dataset):
  - F, Q, R shapes and structure
  - EKF knee angle is more accurate than angles from the noisy keypoints
  - bone lengths converge to the truth
  - coasting through a dropout: uncertainty grows, filter recovers
  - a 60 px outlier on one knee is gated

Run:  python -m pytest tests/test_ekf.py -v
"""

import numpy as np
import pytest

from ekf import skeleton as sk
from ekf.ekf import PoseEKF
from ekf.models import EKFParams, make_F, make_Q, make_R, measurement_std

T, PERIOD, NOISE_PX = 120, 40, 3.0


def true_states() -> np.ndarray:
    """(T, 29) smooth side-view squat: hips drop, torso leans, knees bend, arms rise."""
    t = np.arange(T)
    s = (1 - np.cos(2 * np.pi * t / PERIOD)) / 2            # 0 standing -> 1 bottom
    X = np.zeros((T, sk.N_X))
    X[:, 0], X[:, 1] = 200.0, 300.0 + 40 * s
    X[:, sk.ANG_TORSO] = -np.pi / 2 + 0.5 * s
    for th, sh in ((sk.ANG_THIGH_L, sk.ANG_SHANK_L), (sk.ANG_THIGH_R, sk.ANG_SHANK_R)):
        X[:, th] = np.pi / 2 - 1.2 * s
        X[:, sh] = np.pi / 2 + 0.6 * s
    for ua, fa in ((sk.ANG_UARM_L, sk.ANG_FARM_L), (sk.ANG_UARM_R, sk.ANG_FARM_R)):
        X[:, ua] = np.pi / 2 - 1.4 * s
        X[:, fa] = np.pi / 2 - 1.4 * s
    X[:, sk.LENGTHS] = [60, 3, 3, 35, 30, 50, 50]
    X[:, sk.VEL_P0] = np.gradient(X[:, sk.P0], axis=0)
    X[:, sk.VEL_ANG] = np.gradient(X[:, sk.ANGLES], axis=0)
    return X


def noisy_keypoints(X: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    kp = np.stack([sk.fk(x).reshape(sk.N_KP, 2) for x in X])
    kp = kp + rng.normal(0, NOISE_PX, kp.shape)
    conf = np.full((T, sk.N_KP), 0.9)
    return kp, conf


def knee_from_kp(kp):
    th = sk.seg_angle(kp[:, sk.L_HIP], kp[:, sk.L_KN])
    sh = sk.seg_angle(kp[:, sk.L_KN], kp[:, sk.L_AN])
    return np.abs(sk.wrap(th - sh))


def knee_from_x(X):
    return np.abs(sk.wrap(X[:, sk.ANG_THIGH_L] - X[:, sk.ANG_SHANK_L]))


# ------------------------------------------------------------------ models
def test_F_structure():
    F = make_F(1.0)
    assert F.shape == (sk.N_X, sk.N_X)
    assert F[sk.ANG_THIGH_L, sk.ANG_THIGH_L + sk.VEL_OFFSET] == 1.0
    assert F[sk.LEN_THIGH, sk.LEN_THIGH] == 1.0 and F[sk.LEN_THIGH].sum() == 1.0


def test_Q_symmetric_psd():
    Q = make_Q(EKFParams())
    np.testing.assert_allclose(Q, Q.T)
    assert np.linalg.eigvalsh(Q).min() > -1e-12
    assert Q[sk.ANG_TORSO, sk.ANG_TORSO] == pytest.approx(0.03 ** 2 / 4)


def test_adaptive_R():
    p = EKFParams()
    sig = measurement_std(np.array([1.0, 0.5] + [1.0] * 10), p)
    assert sig[0] == pytest.approx(p.sigma0)
    assert sig[1] == pytest.approx(p.sigma0 * np.sqrt(3.0))
    R = make_R(sig, np.array([True, True] + [False] * 10))
    assert R.shape == (4, 4) and R[2, 2] == pytest.approx(sig[1] ** 2)
    fixed = measurement_std(np.array([0.5] * 12), EKFParams(adaptive_R=False))
    assert np.all(fixed == p.sigma0)


# ------------------------------------------------------------------ filter
@pytest.fixture(scope="module")
def run_clean():
    X = true_states()
    kp, conf = noisy_keypoints(X)
    return X, kp, conf, PoseEKF(EKFParams()).run(kp, conf)


def test_ekf_beats_raw_knee_angle(run_clean):
    X, kp, _, out = run_clean
    warm = 10
    raw = np.degrees(np.sqrt(np.mean((knee_from_kp(kp)[warm:] - knee_from_x(X)[warm:]) ** 2)))
    ekf = np.degrees(np.sqrt(np.mean((knee_from_x(out["x"])[warm:] - knee_from_x(X)[warm:]) ** 2)))
    print(f"knee RMSE  raw {raw:.2f} deg   EKF {ekf:.2f} deg")
    assert ekf < 0.8 * raw                               # about 0.6 with a correct filter


def test_lengths_converge(run_clean):
    X, _, _, out = run_clean
    for l in (sk.LEN_TORSO, sk.LEN_THIGH, sk.LEN_SHANK, sk.LEN_UARM):
        assert out["x"][-1, l] == pytest.approx(X[-1, l], abs=3.0)


def test_nis_consistent_when_R_matches_noise():
    """With R equal to the true keypoint noise, mean NIS / dof should be close to 1.

    (With the default adaptive R at conf 0.9, sigma_j = 4.7 px > the true 3 px, so NIS / dof
    drops to (3 / 4.7)^2 = 0.4: the filter is over-cautious. Same diagnosis as E3.)
    """
    X = true_states()
    kp, conf = noisy_keypoints(X)
    out = PoseEKF(EKFParams(sigma0=NOISE_PX, adaptive_R=False)).run(kp, conf)
    ok = out["status"] == "update"
    ratio = np.nanmean(out["nis"][ok]) / np.mean(out["dof"][ok])
    print(f"mean NIS / dof = {ratio:.2f}  (1.0 = consistent)")
    assert 0.6 < ratio < 1.4


def test_coasting_through_dropout():
    X = true_states()
    kp, conf = noisy_keypoints(X)
    conf[50:65] = 0.0                                    # nobody detected for 15 frames
    out = PoseEKF(EKFParams()).run(kp, conf)
    assert np.all(out["status"][50:65] == "coast")
    var = out["P_diag"][:, sk.ANG_THIGH_L]
    assert var[64] > 5 * var[49]                         # uncertainty grows while blind
    assert var[80] < var[64]                             # and shrinks after recovery
    assert np.all(np.isfinite(out["x"][5:]))


def test_outlier_is_gated():
    X = true_states()
    kp, conf = noisy_keypoints(X)
    kp[70, sk.L_KN] += [60.0, 0.0]                       # confident but wrong knee
    out = PoseEKF(EKFParams()).run(kp, conf)
    assert out["n_gated"][70] >= 1
    out_ng = PoseEKF(EKFParams(gating=False)).run(kp, conf)
    err = lambda o: abs(knee_from_x(o["x"])[70] - knee_from_x(X)[70])
    assert err(out) < err(out_ng)                        # gating protects the estimate
