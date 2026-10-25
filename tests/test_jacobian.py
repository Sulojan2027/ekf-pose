"""
tests/test_jacobian.py

Checks the skeleton model:
  - analytic Jacobian == central finite differences on random states (never drop)
  - y-down convention: an upright standing state has head-to-toe increasing y
  - wrap(), joint limits, length clamping

Run:  python -m pytest tests/test_jacobian.py -v
"""

import numpy as np
import pytest

from ekf import skeleton as sk


def random_state(rng: np.random.Generator) -> np.ndarray:
    """A random but plausible 29-D state (any angles, positive lengths)."""
    x = np.zeros(sk.N_X)
    x[sk.P0] = rng.uniform(100, 400, 2)
    x[sk.ANGLES] = rng.uniform(-np.pi, np.pi, sk.N_ANG)
    x[sk.VEL_P0] = rng.normal(0, 2, 2)
    x[sk.VEL_ANG] = rng.normal(0, 0.05, sk.N_ANG)
    x[sk.LENGTHS] = rng.uniform(10, 80, 7)
    return x


def standing_state() -> np.ndarray:
    """Upright person: torso up (-pi/2), all limbs hanging down (+pi/2)."""
    x = np.zeros(sk.N_X)
    x[sk.P0] = [200.0, 300.0]
    x[sk.ANGLES] = np.pi / 2
    x[sk.ANG_TORSO] = -np.pi / 2
    x[sk.LENGTHS] = [60, 20, 12, 35, 30, 50, 50]
    return x


def numerical_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference Jacobian of f at x.

    Column k:  (f(x + eps e_k) - f(x - eps e_k)) / (2 eps)     error O(eps^2)
    Returns (len(f(x)), len(x)).
    """
    # TODO: loop over k, perturb a COPY of x in both directions, stack the columns
    #       (np.column_stack or fill a preallocated array).
    raise NotImplementedError


@pytest.mark.parametrize("seed", range(20))
def test_jacobian_matches_finite_differences(seed):
    x = random_state(np.random.default_rng(seed))
    H = sk.jacobian(x)
    Hn = numerical_jacobian(sk.fk, x)
    assert H.shape == (sk.N_Z, sk.N_X)
    np.testing.assert_allclose(H, Hn, rtol=1e-5, atol=1e-6)


def test_velocity_columns_are_zero():
    H = sk.jacobian(random_state(np.random.default_rng(0)))
    assert np.all(H[:, sk.VEL_P0] == 0) and np.all(H[:, sk.VEL_ANG] == 0)


def test_standing_pose_y_down():
    z = sk.fk(standing_state()).reshape(sk.N_KP, 2)
    y = z[:, 1]
    for side in ("L", "R"):
        sh, hip, kn, an = (getattr(sk, f"{side}_{n}") for n in ("SH", "HIP", "KN", "AN"))
        assert y[sh] < y[hip] < y[kn] < y[an], f"{side}: shoulder above hip above knee above ankle"
    # left and right keypoints mirror around the root in x
    assert z[sk.L_HIP, 0] - 200 == pytest.approx(-(z[sk.R_HIP, 0] - 200))


def test_fk_inverts_init():
    """IK init on fk(x) should give back fk(x) (lengths/angles recovered)."""
    x = standing_state()
    x[sk.ANG_THIGH_L] = 0.3          # bend things a bit
    x[sk.ANG_SHANK_L] = 2.0
    x[sk.ANG_THIGH_R] = 0.3
    x[sk.ANG_SHANK_R] = 2.0
    kp = sk.fk(x).reshape(sk.N_KP, 2)
    x0, P0 = sk.init_from_keypoints(kp, np.ones(sk.N_KP, bool))
    np.testing.assert_allclose(sk.fk(x0), sk.fk(x), atol=1e-6)
    assert P0.shape == (sk.N_X, sk.N_X)


def test_wrap_range():
    a = np.linspace(-10, 10, 1001)
    w = sk.wrap(a)
    assert np.all(w > -np.pi - 1e-12) and np.all(w <= np.pi + 1e-12)
    np.testing.assert_allclose(np.sin(w), np.sin(a), atol=1e-12)
    np.testing.assert_allclose(np.cos(w), np.cos(a), atol=1e-12)


def test_joint_limits_clamp_knee_only_when_violated():
    x = standing_state()
    x[sk.ANG_THIGH_L], x[sk.ANG_SHANK_L] = 0.0, np.deg2rad(-175)   # rel = 175 deg > 165
    x[sk.ANG_THIGH_R], x[sk.ANG_SHANK_R] = 0.0, np.deg2rad(-90)    # rel = 90 deg, fine
    y = sk.apply_joint_limits(x)
    knee_L = abs(sk.wrap(y[sk.ANG_THIGH_L] - y[sk.ANG_SHANK_L]))
    assert knee_L == pytest.approx(sk.KNEE_LIMIT)
    assert y[sk.ANG_THIGH_L] == x[sk.ANG_THIGH_L]                   # proximal untouched
    assert y[sk.ANG_SHANK_R] == x[sk.ANG_SHANK_R]                   # no violation, no change


def test_clamp_lengths():
    x = standing_state()
    x[sk.LENGTHS] = [-3, -1, 0.5, 2, 100, 4, 7]
    y = sk.clamp_lengths(x)
    assert y[sk.LEN_SHW] == sk.MIN_WIDTH and y[sk.LEN_HIPW] == sk.MIN_WIDTH
    assert y[sk.LEN_TORSO] == sk.MIN_LENGTH and y[sk.LEN_UARM] == sk.MIN_LENGTH
    assert y[sk.LEN_FARM] == 100 and y[sk.LEN_SHANK] == 7
