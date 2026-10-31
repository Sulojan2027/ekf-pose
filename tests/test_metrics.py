"""
tests/test_metrics.py

Known-answer tests for eval/metrics.py, baselines/ and sim/simulate.py.
Each test builds an input whose correct answer can be worked out by hand.

Run:  python -m pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from baselines.kf_per_keypoint import kf_per_keypoint
from baselines.moving_average import moving_average
from ekf import skeleton as sk
from ekf.models import EKFParams
from eval import metrics as M
from sim.simulate import SimConfig, cv_states, observe, squat_states

T = 60


# ------------------------------------------------------------------ eval/metrics.py
def test_angle_rmse():
    gt = np.linspace(0.2, 1.5, T)
    est = gt + 0.1
    valid = np.ones(T, bool)
    assert M.angle_rmse(est, gt, valid) == pytest.approx(np.degrees(0.1))
    est[5], valid[6] = np.nan, False                     # NaN and invalid frames are skipped
    est[6] = 99.0
    assert M.angle_rmse(est, gt, valid) == pytest.approx(np.degrees(0.1))
    assert np.isnan(M.angle_rmse(est, gt, np.zeros(T, bool)))


def test_jitter():
    assert M.jitter(np.linspace(0, 1, T)) == pytest.approx(0.0, abs=1e-9)   # a ramp is not shaky
    a = 0.01 * (-1.0) ** np.arange(T)                    # zig-zag: |a''| = 4 * 0.01 everywhere
    assert M.jitter(a) == pytest.approx(np.degrees(0.04))
    a[10] = np.nan                                       # triples touching a NaN are skipped
    assert M.jitter(a) == pytest.approx(np.degrees(0.04))


def test_lag():
    t = np.arange(200)
    gt = np.sin(2 * np.pi * t / 40)
    valid = np.ones(200, bool)
    late = np.roll(gt, 3); late[:3] = np.nan             # est[t] = gt[t - 3]
    assert M.lag(late, gt, valid) == 3
    early = np.roll(gt, -2); early[-2:] = np.nan
    assert M.lag(early, gt, valid) == -2
    assert M.lag(gt, gt, valid) == 0


def test_keypoint_rmse_pct():
    gt = np.random.default_rng(0).uniform(0, 300, (T, 12, 2))
    est = gt + np.array([3.0, 4.0])                      # every error is 5 px
    vis = np.ones((T, 12), bool)
    bbox = np.tile([0.0, 0.0, 50.0, 100.0], (T, 1))      # box height 100 px
    assert M.keypoint_rmse_pct(est, gt, vis, bbox) == pytest.approx(5.0)
    est[:, 0] += 1000.0; vis[:, 0] = False               # invisible joints are ignored
    assert M.keypoint_rmse_pct(est, gt, vis, bbox) == pytest.approx(5.0)


def test_bone_length_cv():
    x = np.zeros(sk.N_X); x[sk.P0] = [200, 300]; x[sk.ANGLES] = np.pi / 2; x[sk.ANG_TORSO] = -np.pi / 2
    x[sk.LENGTHS] = [60, 3, 3, 35, 30, 50, 50]
    kp = []
    for k in range(T):                                   # rigid leg moving: CV must be 0
        x[sk.ANG_THIGH_L] = np.pi / 2 - 0.02 * k
        kp.append(sk.fk(x).reshape(12, 2))
    kp = np.array(kp)
    assert M.bone_length_cv(kp, "L") == pytest.approx(0.0, abs=1e-9)
    kp[::2, sk.L_KN] = kp[::2, sk.L_HIP] + 1.1 * (kp[::2, sk.L_KN] - kp[::2, sk.L_HIP])  # thigh 50 / 55
    # shank also changes (knee moved), so just check it is clearly > 0 and finite
    assert 1.0 < M.bone_length_cv(kp, "L") < 20.0


def test_nees():
    err = np.array([[2.0, 0.0], [0.0, 3.0]])
    P = np.stack([4 * np.eye(2), np.diag([1.0, 9.0])])
    np.testing.assert_allclose(M.nees(err, P), [1.0, 1.0])


def test_chi2_band():
    lo, hi = M.chi2_band(dof=2, n_runs=1)
    assert lo == pytest.approx(0.0506, abs=1e-3) and hi == pytest.approx(7.378, abs=1e-3)
    lo, hi = M.chi2_band(dof=29, n_runs=30)
    assert lo < 29 < hi and hi - lo < 6                  # averaging 30 runs narrows the band


# ------------------------------------------------------------------ baselines
def test_moving_average():
    kp = np.zeros((T, 12, 2)); kp[..., 0] = np.arange(T)[:, None]       # x = t (a ramp)
    conf = np.ones((T, 12))
    np.testing.assert_allclose(moving_average(kp, conf, window=1), kp)
    out = moving_average(kp, conf, window=3)
    assert out[10, 0, 0] == pytest.approx(9.0)          # mean of 8, 9, 10: causal -> 1 frame lag
    assert out[0, 0, 0] == pytest.approx(0.0)           # shorter window at the start
    conf[20:25, 3] = 0.0; kp[20:25, 3] = 0.0           # undetected: YOLO zeros must NOT pull it
    out = moving_average(kp, conf, window=3)
    assert out[21, 3, 0] == pytest.approx(19.0)         # only frame 19 is valid in [19, 21]
    assert np.isnan(out[23, 3, 0])                       # no valid sample in [21, 23]


def test_kf_per_keypoint():
    rng = np.random.default_rng(0)
    t = np.arange(T)
    truth = np.zeros((T, 12, 2)); truth[..., 0] = 100 + 2.0 * t[:, None]; truth[..., 1] = 50.0
    kp = truth + rng.normal(0, 2.0, truth.shape)
    conf = np.ones((T, 12)); conf[:5, 0] = 0.0; conf[30:40, 1] = 0.0
    out, var = kf_per_keypoint(kp, conf, EKFParams(sigma0=2.0, adaptive_R=False), sigma_a=0.5)
    assert np.all(np.isnan(out[:5, 0])) and np.all(np.isfinite(out[5:, 0]))   # waits for 1st detection
    err_raw = np.abs(kp[20:] - truth[20:]).mean()
    err_kf = np.abs(out[20:, 2:] - truth[20:, 2:]).mean()
    assert err_kf < 0.7 * err_raw                       # smoother than the input
    assert var[39, 1] > 3 * var[29, 1]                  # uncertainty grows while coasting


# ------------------------------------------------------------------ sim
def test_observe_noise_matches_model():
    X = np.repeat(squat_states(10, 45, np.random.default_rng(0))[:1], 4000, axis=0)
    cfg = SimConfig(sigma0=3.0, alpha=4.0, conf_fixed=0.5)
    kp, conf, clean = observe(X, cfg, np.random.default_rng(1))
    std = (kp - clean).std()
    assert std == pytest.approx(3.0 * np.sqrt(1 + 4.0 * 0.5), rel=0.03)      # 5.196 px
    assert np.all(conf == 0.5)


def test_cv_states_follow_model():
    p = EKFParams(sigma_a=0.5, sigma_alpha=0.01, sigma_L=0.02)
    x0 = squat_states(10, 45, np.random.default_rng(0))[0]
    x0[sk.VEL_P0] = 0; x0[sk.VEL_ANG] = 0
    X = cv_states(x0, p, 2000, np.random.default_rng(2))
    np.testing.assert_allclose(X[0], x0)
    # velocity increments are the acceleration noise:  std(diff(v)) = sigma * dt
    assert np.diff(X[:, sk.VEL_P0.start]).std() == pytest.approx(0.5, rel=0.1)
    assert np.diff(X[:, sk.LEN_THIGH]).std() == pytest.approx(0.02, rel=0.1)
    assert np.all(np.abs(X[:, sk.ANGLES]) <= np.pi + 1e-12)                  # wrapped
