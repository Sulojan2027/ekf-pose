"""
ekf/skeleton.py

2D planar skeleton model for the EKF:
  - state index constants (29-D state)        [filled: transcription of HANDOFF 7.2]
  - segment table + kinematic chains           [segments filled, chains partly TODO]
  - fk(x):        forward kinematics h(x) -> 12 keypoints (24-D z)
  - jacobian(x):  analytic H = dh/dx (24 x 29)
  - init_from_keypoints(): inverse-kinematics initialisation (x0, P0)
  - joint limits, length clamping, angle helpers, flexion angle definitions

Conventions (HANDOFF 7.1):
  image coords, x right, y DOWN. Absolute angles theta = atan2(dy, dx) in image
  coords, so a positive angle is CLOCKWISE on screen. Upright torso
  (shoulders above hips) has theta_torso ~ -pi/2; a hanging limb has ~ +pi/2.
  u(t)  = [cos t, sin t]
  u_perp(t) = u(t + pi/2) = [-sin t, cos t]
  All angles and relative angles are wrapped to (-pi, pi].

Keypoint order (local 0..11) == COCO 5..16 == Penn 1..12:
  0 L_sh, 1 R_sh, 2 L_el, 3 R_el, 4 L_wr, 5 R_wr,
  6 L_hip, 7 R_hip, 8 L_kn, 9 R_kn, 10 L_an, 11 R_an
Measurement z = [x0, y0, x1, y1, ..., x11, y11]  -> keypoint j is rows 2j, 2j+1.
"""

import numpy as np

# ----------------------------------------------------------------------------
# State layout (29-D)
# ----------------------------------------------------------------------------
# root position (mid-hip), px
P0 = slice(0, 2)

# absolute segment angles, rad
ANG_TORSO = 2     # hip -> shoulder midpoint
ANG_UARM_L = 3    # shoulder -> elbow
ANG_FARM_L = 4    # elbow -> wrist
ANG_UARM_R = 5
ANG_FARM_R = 6
ANG_THIGH_L = 7   # hip -> knee
ANG_SHANK_L = 8   # knee -> ankle
ANG_THIGH_R = 9
ANG_SHANK_R = 10
ANGLES = slice(2, 11)
N_ANG = 9

# velocities: root (px/frame) then the 9 angles (rad/frame), same order
VEL_P0 = slice(11, 13)
VEL_ANG = slice(13, 22)
VEL_OFFSET = 11          # velocity of state i (i in 0..10) is at i + VEL_OFFSET

# lengths, px (left/right share one length)
LEN_TORSO = 22    # hip midpoint -> shoulder midpoint
LEN_SHW = 23      # shoulder HALF-width (mid-shoulder -> each shoulder)
LEN_HIPW = 24     # hip HALF-width (mid-hip -> each hip)
LEN_UARM = 25
LEN_FARM = 26
LEN_THIGH = 27
LEN_SHANK = 28
LENGTHS = slice(22, 29)
WIDTHS = (LEN_SHW, LEN_HIPW)

N_X = 29
N_KP = 12
N_Z = 2 * N_KP

# local keypoint indices
KP_NAMES = ["L_sh", "R_sh", "L_el", "R_el", "L_wr", "R_wr",
            "L_hip", "R_hip", "L_kn", "R_kn", "L_an", "R_an"]
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = range(N_KP)
COCO_BODY = slice(5, 17)          # coco[:, 5:17] -> local 0..11

# clamping / limits
MIN_WIDTH = 1.0                   # px, W_s and W_h
MIN_LENGTH = 5.0                  # px, all other lengths
KNEE_LIMIT = np.deg2rad(165.0)    # |wrap(thigh - shank)|
ELBOW_LIMIT = np.deg2rad(160.0)   # |wrap(uarm - farm)|

# anthropometric length ratios w.r.t. torso length (Winter, segment lengths /
# body height: torso~0.30, thigh~0.245, shank~0.246, uarm~0.186, farm~0.146)
DEFAULT_RATIO = {
    LEN_UARM: 0.62, LEN_FARM: 0.49, LEN_THIGH: 0.82, LEN_SHANK: 0.82,
    LEN_SHW: 0.10, LEN_HIPW: 0.10,   # small: side views, width ~ 0
}

# initial covariance std-devs (HANDOFF 7.8)
P0_STD_POS = 10.0      # px
P0_STD_ANG = 0.2       # rad
P0_STD_VPOS = 10.0     # px/frame   ("large": velocity unknown at t=0)
P0_STD_VANG = 0.3      # rad/frame
P0_STD_LEN = 5.0       # px

# ----------------------------------------------------------------------------
# Segment table
# ----------------------------------------------------------------------------
# A segment is (angle_idx, length_idx, sign, perp):
#   contribution to a keypoint = sign * x[length_idx] * d
#   d = u(x[angle_idx])        if not perp
#   d = u_perp(x[angle_idx])   if perp   (width segments, perpendicular to torso)
SEG_TORSO = (ANG_TORSO, LEN_TORSO, +1, False)
SEG_SHW_L = (ANG_TORSO, LEN_SHW, +1, True)
SEG_SHW_R = (ANG_TORSO, LEN_SHW, -1, True)
SEG_HIPW_L = (ANG_TORSO, LEN_HIPW, +1, True)
SEG_HIPW_R = (ANG_TORSO, LEN_HIPW, -1, True)
SEG_UARM_L = (ANG_UARM_L, LEN_UARM, +1, False)
SEG_FARM_L = (ANG_FARM_L, LEN_FARM, +1, False)
SEG_UARM_R = (ANG_UARM_R, LEN_UARM, +1, False)
SEG_FARM_R = (ANG_FARM_R, LEN_FARM, +1, False)
SEG_THIGH_L = (ANG_THIGH_L, LEN_THIGH, +1, False)
SEG_SHANK_L = (ANG_SHANK_L, LEN_SHANK, +1, False)
SEG_THIGH_R = (ANG_THIGH_R, LEN_THIGH, +1, False)
SEG_SHANK_R = (ANG_SHANK_R, LEN_SHANK, +1, False)

# Kinematic chain of each keypoint: keypoint = p0 + sum of its segments (root -> tip).
CHAINS = {
    L_SH: [SEG_TORSO, SEG_SHW_L],
    L_KN: [SEG_HIPW_L, SEG_THIGH_L],
    R_SH: [SEG_TORSO, SEG_SHW_R],
    R_KN: [SEG_HIPW_R, SEG_THIGH_R],
    L_WR: [SEG_TORSO, SEG_SHW_L, SEG_UARM_L, SEG_FARM_L],
    R_WR: [SEG_TORSO, SEG_SHW_R, SEG_UARM_R, SEG_FARM_R],
    L_EL: [SEG_TORSO, SEG_SHW_L, SEG_UARM_L],
    R_EL: [SEG_TORSO, SEG_SHW_R, SEG_UARM_R],
    L_HIP: [SEG_HIPW_L],
    R_HIP: [SEG_HIPW_R],
    L_AN: [SEG_HIPW_L, SEG_THIGH_L, SEG_SHANK_L],
    R_AN: [SEG_HIPW_R, SEG_THIGH_R, SEG_SHANK_R]
}

# ----------------------------------------------------------------------------
# Angle helpers
# ----------------------------------------------------------------------------
def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))

def seg_dir(theta: float, perp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction of a segment and its derivative w.r.t. theta.

    Returns:
        d:  (2,)  u(theta) or u_perp(theta)
        dd: (2,)  d d / d theta
    Math (use it in the report):
        u(t)      = [ cos t,  sin t]   ->  d/dt = [-sin t,  cos t] =  u_perp(t)
        u_perp(t) = [-sin t,  cos t]   ->  d/dt = [-cos t, -sin t] = -u(t)
        In both cases dd = d rotated by +90 deg:  dd = [-d_y, d_x].
    """
    if perp:
        d = np.array([-np.sin(theta), np.cos(theta)])
    else:
        d = np.array([np.cos(theta), np.sin(theta)])
    dd = np.array([-d[1], d[0]])
    return d, dd

def seg_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute angle of the vector a -> b.  a, b: (..., 2) -> (...,) rad."""
    d = np.asarray(b) - np.asarray(a)
    return np.arctan2(d[..., 1], d[..., 0])

# ----------------------------------------------------------------------------
# Measurement model
# ----------------------------------------------------------------------------
def fk(x: np.ndarray) -> np.ndarray:
    """Forward kinematics h(x).

    Args:
        x: (29,) state.
    Returns:
        z: (24,) predicted keypoints [x0, y0, ..., x11, y11] in px.
    Equation:  k_j = p0 + sum_{(a, l, s, perp) in CHAINS[j]} s * x[l] * d(x[a], perp)
    """
    z = np.zeros(N_Z)
    for j in range(N_KP):
        pos = x[P0].copy()
        for (a, l, s, perp) in CHAINS[j]:
            pos += s * x[l] * seg_dir(x[a], perp)[0]
        z[2 * j: 2 * j + 2] = pos
    return z


def jacobian(x: np.ndarray) -> np.ndarray:
    """Analytic Jacobian H = dh/dx at x.

    Args:
        x: (29,) state.
    Returns:
        H: (24, 29).
    Derivation (k_j = p0 + sum_i s_i l_i d_i(theta_{a_i})):
        dk_j/dp0         = I_2                          -> columns P0
        dk_j/dtheta_a   += s_i l_i dd_i  for every segment i of chain j with angle a
        dk_j/dl_l       += s_i d_i       for every segment i of chain j with length l
        dk_j/dvelocities = 0             (h does not depend on velocities)
    IMPORTANT: use +=, not =. theta_torso appears in TWO segments of the shoulder
    chains (torso length along u, half-width along u_perp), so both contributions add:
        d shoulder_L / d theta_t = L_t u_perp(theta_t) + W_s * (-u(theta_t))
    """
    H = np.zeros((N_Z, N_X))
    for j in range(N_KP):
        r = slice(2 * j, 2 * j + 2)
        H[r, P0] = np.eye(2)
        for (a, l, s, perp) in CHAINS[j]:
            d, dd = seg_dir(x[a], perp)
            H[r, a] += s * x[l] * dd
            H[r, l] += s * d
    return H


# ----------------------------------------------------------------------------
# Constraints applied after each update
# ----------------------------------------------------------------------------
def clamp_lengths(x: np.ndarray) -> np.ndarray:
    """Return a copy of x with widths >= MIN_WIDTH and other lengths >= MIN_LENGTH."""
    # TODO: x = x.copy(); clip x[LENGTHS] at MIN_LENGTH, then set the two WIDTHS
    #       entries with max(MIN_WIDTH, value) (widths are allowed to be small).
    #       Careful: do the widths from the ORIGINAL values, not after the 5 px clip.
    x = x.copy()
    x[LENGTHS] = np.maximum()


def wrap_angles(x: np.ndarray) -> np.ndarray:
    """Return a copy of x with the 9 absolute angles wrapped to (-pi, pi]."""
    x = x.copy()
    x[ANGLES] = wrap(x[ANGLES])
    return x


def apply_joint_limits(x: np.ndarray) -> np.ndarray:
    """Sign-agnostic joint-limit projection (HANDOFF 7.7). Returns a copy.

    For each (proximal, distal, limit) in
        (THIGH_L, SHANK_L, KNEE), (THIGH_R, SHANK_R, KNEE),
        (UARM_L, FARM_L, ELBOW), (UARM_R, FARM_R, ELBOW):
      rel = wrap(theta_prox - theta_dist)
      if |rel| > limit:  theta_dist = wrap(theta_prox - sign(rel) * limit)
    i.e. move only the DISTAL segment back onto the limit, keep its sign.
    This is a projection of the estimate onto the feasible set (P is left as is).
    """
    for prox, dist, limit in [
        (ANG_THIGH_L, ANG_SHANK_L, KNEE_LIMIT),
        (ANG_THIGH_R, ANG_SHANK_R, KNEE_LIMIT),
        (ANG_UARM_L, ANG_FARM_L, ELBOW_LIMIT),
        (ANG_UARM_R, ANG_FARM_R, ELBOW_LIMIT)
    ]:
        rel = wrap(x[prox] - x[dist])
        if abs(rel) > limit:
            x[dist] = wrap(x[prox] - np.sign(rel) * limit)
    return x

# ----------------------------------------------------------------------------
# Evaluation angle definitions (used identically for GT, raw, baselines, EKF)
# ----------------------------------------------------------------------------
def flexion_angles(th_torso, th_thigh, th_shank) -> tuple[np.ndarray, np.ndarray]:
    """Knee and hip flexion magnitudes (rad), 0 when standing straight.

    knee = |wrap(th_thigh - th_shank)|          straight leg -> 0
    hip  = pi - |wrap(th_torso - th_thigh)|     torso up (-pi/2) + thigh down (+pi/2)
                                                -> difference -pi -> hip = 0
    Sign-agnostic, so left- and right-facing clips give the same numbers.
    Inputs: arrays of the same shape (or scalars). Outputs: same shape.
    """
    # TODO: two lines, using wrap().
    knee = np.abs(wrap(th_thigh - th_shank))
    hip = np.pi - np.abs(wrap(th_torso - th_thigh))
    return knee, hip

# ----------------------------------------------------------------------------
# Initialisation (inverse kinematics)
# ----------------------------------------------------------------------------
def init_from_keypoints(kp: np.ndarray, ok: np.ndarray):
    """IK initialisation from one frame of keypoints.

    Args:
        kp: (12, 2) keypoints in local order (px).
        ok: (12,) bool, keypoint usable (conf >= c_min).
    Returns:
        (x0 (29,), P0 (29, 29)) or None if the frame is not usable
        (need >= 1 hip and >= 1 shoulder; caller then tries the next frame).

    Steps:
      1. p0     = mean of the usable hips;  mid_sh = mean of the usable shoulders.
      2. theta_t = seg_angle(p0, mid_sh);   L_t = |mid_sh - p0|.
      3. widths: project onto the torso normal so the sign convention is respected
             W_s = dot(sh_L - mid_sh, u_perp(theta_t))  if both shoulders usable
             W_h = dot(hip_L - p0,    u_perp(theta_t))  if both hips usable
         otherwise DEFAULT_RATIO * L_t. (clamp_lengths at the end makes them >= 1.)
      4. for each limb segment (uarm, farm, thigh, shank), side L and R:
             angle  = seg_angle(proximal kp, distal kp) if both usable
             length = distance                          if both usable
         length per segment type = mean of the available sides, else DEFAULT_RATIO * L_t;
         angle of a missing side = angle of the other side if available, else +pi/2 (hanging).
      5. velocities = 0.
      6. P0 = diag of squared stds: P0_STD_POS (p0), P0_STD_ANG (angles),
         P0_STD_VPOS (root vel), P0_STD_VANG (angle vel), P0_STD_LEN (lengths).
      7. return clamp_lengths(wrap_angles(x0)), P0.
    Tip: a table [(ANG_x, LEN_x, prox_kp, dist_kp), ...] for the 8 limb segments
         keeps step 4 short.
    """
    hips = [kp[j] for j in (L_HIP, R_HIP) if ok[j]]

def split_coco(kp17: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(..., 17, 3) YOLO/COCO keypoints -> (..., 12, 2) positions, (..., 12) conf.   [glue]"""
    body = kp17[..., COCO_BODY, :]
    return body[..., :2], body[..., 2]
