"""
ekf/skeleton.py
"""

import numpy as np

# --------------------
# State layout (29-D)
# --------------------
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

# velocities: root (px/frame) then the 9 angles (rad/frame)
VEL_P0 = slice(11, 13)
VEL_ANG = slice(13, 22)
VEL_OFFSET = 11          

# lengths, px (left/right share one length)
LEN_TORSO = 22   
LEN_SHW = 23     
LEN_HIPW = 24     
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
MIN_WIDTH = 1.0                 
MIN_LENGTH = 5.0                  
KNEE_LIMIT = np.deg2rad(165.0)  
ELBOW_LIMIT = np.deg2rad(160.0)  

DEFAULT_RATIO = {
    LEN_UARM: 0.62, LEN_FARM: 0.49, LEN_THIGH: 0.82, LEN_SHANK: 0.82,
    LEN_SHW: 0.10, LEN_HIPW: 0.10,   # small: side views, width ~ 0
}

# initial covariance std-devs
P0_STD_POS = 10.0      
P0_STD_ANG = 0.2       
P0_STD_VPOS = 10.0     
P0_STD_VANG = 0.3    
P0_STD_LEN = 5.0      

# ---------------
# Segment table
# ---------------
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

# Kinematic chain of each keypoint
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

# --------------
# Angle helpers
# --------------
def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))

def seg_dir(theta: float, perp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction of a segment and its derivative w.r.t. theta.
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


# --------------------------------------
# Constraints applied after each update
# --------------------------------------
def clamp_lengths(x: np.ndarray) -> np.ndarray:
    """Return a copy of x with widths >= MIN_WIDTH and other lengths >= MIN_LENGTH."""
    x = x.copy()
    widths = x[list(WIDTHS)].copy()
    x[LENGTHS] = np.maximum(x[LENGTHS], MIN_LENGTH)
    x[list(WIDTHS)] = np.maximum(widths, MIN_WIDTH)
    return x

def wrap_angles(x: np.ndarray) -> np.ndarray:
    """Return a copy of x with the 9 absolute angles wrapped to (-pi, pi]."""
    x = x.copy()
    x[ANGLES] = wrap(x[ANGLES])
    return x


def apply_joint_limits(x: np.ndarray) -> np.ndarray:
    """Sign-agnostic joint-limit projection (HANDOFF 7.7). Returns a copy.
    """
    x = x.copy()
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
    """
    knee = np.abs(wrap(th_thigh - th_shank))
    hip = np.pi - np.abs(wrap(th_torso - th_thigh))
    return knee, hip

# ----------------------------------------------------------------------------
# Initialisation (inverse kinematics)
# ----------------------------------------------------------------------------
def init_from_keypoints(kp: np.ndarray, ok: np.ndarray):
    """IK initialisation from one frame of keypoints.]
         keeps step 4 short.
    """
    # usable hips and shoulders
    hips = [kp[j] for j in (L_HIP, R_HIP) if ok[j]]
    shoulders = [kp[j] for j in (L_SH, R_SH) if ok[j]]
    if not hips or not shoulders:
        return None

    x0 = np.zeros(N_X)

    # root and torso
    p0 = np.mean(hips, axis=0)
    mid_sh = np.mean(shoulders, axis=0)
    theta_t = seg_angle(p0, mid_sh)
    L_t = np.linalg.norm(mid_sh - p0)
    x0[P0] = p0
    x0[ANG_TORSO] = theta_t
    x0[LEN_TORSO] = L_t

    # widths
    n_perp = seg_dir(theta_t, True)[0]
    if ok[L_SH] and ok[R_SH]:
        x0[LEN_SHW] = np.dot(kp[L_SH] - mid_sh, n_perp)
    else:
        x0[LEN_SHW] = DEFAULT_RATIO[LEN_SHW] * L_t
    if ok[L_HIP] and ok[R_HIP]:
        x0[LEN_HIPW] = np.dot(kp[L_HIP] - p0, n_perp)
    else:
        x0[LEN_HIPW] = DEFAULT_RATIO[LEN_HIPW] * L_t

    # limbs, grouped by segment type
    LIMBS = [
        (LEN_UARM,  (ANG_UARM_L,  L_SH,  L_EL), (ANG_UARM_R,  R_SH,  R_EL)),
        (LEN_FARM,  (ANG_FARM_L,  L_EL,  L_WR), (ANG_FARM_R,  R_EL,  R_WR)),
        (LEN_THIGH, (ANG_THIGH_L, L_HIP, L_KN), (ANG_THIGH_R, R_HIP, R_KN)),
        (LEN_SHANK, (ANG_SHANK_L, L_KN,  L_AN), (ANG_SHANK_R, R_KN,  R_AN)),
    ]
    for l, *sides in LIMBS:
        angles, lengths = {}, []          
        for (a, i, k) in sides:
            if ok[i] and ok[k]:
                angles[a] = seg_angle(kp[i], kp[k])
                lengths.append(np.linalg.norm(kp[k]- kp[i]))
        x0[l] = np.mean(lengths) if lengths else DEFAULT_RATIO[l] * L_t

        # angle
        fallback = next(iter(angles.values()), np.pi / 2)
        for (a, _, _) in sides:
            x0[a] = angles.get(a, fallback)

    std = np.concatenate([np.full(2, P0_STD_POS), np.full(9, P0_STD_ANG), np.full(2, P0_STD_VPOS),
                          np.full(9, P0_STD_VANG), np.full(7, P0_STD_LEN)])
    P_init = np.diag(std ** 2)

    return clamp_lengths(wrap_angles(x0)), P_init

def split_coco(kp17: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(..., 17, 3) YOLO/COCO keypoints -> (..., 12, 2) positions, (..., 12) conf.   [glue]"""
    body = kp17[..., COCO_BODY, :]
    return body[..., :2], body[..., 2]
