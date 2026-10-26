"""Numerical interface of InternRobotics/HIMLoco's official Go1 policy.

Source: ef289acaa62795009363b7b819c9186690630441, legged_robot.py,
legged_robot_config.py, go1_config.py and PolicyExporterHIM in helpers.py.
This module intentionally has no LCM, PyTorch, simulator, or camera dependency.
"""

import numpy as np

UPSTREAM_COMMIT = "ef289acaa62795009363b7b819c9186690630441"
OBS_DIM = 45
HISTORY_FRAMES = 6
HISTORY_DIM = OBS_DIM * HISTORY_FRAMES
ACTION_DIM = 12
POLICY_DT = 0.02
ACTION_SCALE = 0.25
KP = 40.0
KD = 1.0
OBS_CLIP = 100.0
ACTION_CLIP = 100.0

# This is SDK motor order, NOT a claim about a trained policy's DOF order.
SDK_JOINT_NAMES = tuple(
    "{}_{}_joint".format(leg, joint)
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
DEFAULT_ANGLES = {
    "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
}
# Source URDF bounds; these are not an experimentally validated safety envelope.
_JOINT_LIMITS = {
    "hip": (-0.802851455917, 0.802851455917),
    "thigh": (-1.0471975512, 4.18879020479),
    "calf": (-2.69653369433, -0.916297857297),
}


def finite_vector(value, length, name):
    """Validate size before conversion; reject nonfinite/overflowed values."""
    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (length,):
        raise ValueError("{} must have shape ({},), got {}".format(name, length, vector.shape))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain only finite float32 values".format(name))
    return vector.copy()


class JointMap:
    """Map by recorded training names, without assuming a self-inverse swap."""

    def __init__(self, policy_names):
        if not isinstance(policy_names, (list, tuple)):
            raise ValueError("training_dof_names must be an explicit ordered list of 12 names")
        if (len(policy_names) != ACTION_DIM or
                any(not isinstance(name, str) for name in policy_names) or
                set(policy_names) != set(SDK_JOINT_NAMES)):
            raise ValueError("training_dof_names must contain every Go1 joint exactly once")
        self.policy_names = tuple(policy_names)
        self.policy_to_sdk = np.asarray([SDK_JOINT_NAMES.index(name) for name in policy_names])
        self.sdk_to_policy = np.argsort(self.policy_to_sdk)

    def to_policy(self, sdk_array):
        return finite_vector(sdk_array, ACTION_DIM, "SDK joint vector")[self.policy_to_sdk].copy()

    def to_sdk(self, policy_array):
        return finite_vector(policy_array, ACTION_DIM, "policy joint vector")[self.sdk_to_policy].copy()


def default_positions(policy_names):
    names = JointMap(policy_names).policy_names
    return np.asarray([DEFAULT_ANGLES[name] for name in names], dtype=np.float32)


def joint_limits(policy_names):
    names = JointMap(policy_names).policy_names
    limits = [_JOINT_LIMITS[name.split("_")[1]] for name in names]
    return (np.asarray([x[0] for x in limits], dtype=np.float32),
            np.asarray([x[1] for x in limits], dtype=np.float32))


def clip_action(action):
    """Training clipping only: +/-100 is NOT a mechanical safety limit."""
    return np.clip(finite_vector(action, ACTION_DIM, "action"), -ACTION_CLIP, ACTION_CLIP)


def action_to_target(action, q0):
    return finite_vector(q0, ACTION_DIM, "default positions") + ACTION_SCALE * clip_action(action)


def build_observation(commands, omega_body, gravity_body, q_policy,
                      dq_policy, previous_action, q0):
    """Build one noise-free, clipped frame in policy DOF order.

    previous_action is the previous unscaled clipped policy action during normal
    operation. If startup blending changes the applied action, its representation
    is an explicitly documented deployment choice made by the runner.
    """
    command = finite_vector(commands, 3, "commands")
    gyro = finite_vector(omega_body, 3, "body gyro")
    gravity = finite_vector(gravity_body, 3, "projected gravity")
    if not np.isclose(np.linalg.norm(gravity), 1.0, atol=1e-4):
        raise ValueError("projected gravity must be a unit vector (not accelerometer m/s^2)")
    q = finite_vector(q_policy, ACTION_DIM, "joint positions")
    dq = finite_vector(dq_policy, ACTION_DIM, "joint velocities")
    previous = finite_vector(previous_action, ACTION_DIM, "previous action")
    nominal = finite_vector(q0, ACTION_DIM, "default positions")
    if np.any(np.abs(previous) > ACTION_CLIP):
        raise ValueError("previous action must already be clipped to the training action range")
    with np.errstate(over="ignore", invalid="ignore"):
        obs = np.concatenate((command * np.asarray([2.0, 2.0, 0.25], dtype=np.float32),
                              gyro * 0.25, gravity, q - nominal, dq * 0.05, previous))
    if not np.all(np.isfinite(obs)):
        raise ValueError("observation arithmetic overflow")
    return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32, copy=False)


class ObservationHistory:
    """Six frames, newest first. Only push() mutates temporal state."""

    def __init__(self):
        self._history = np.zeros((HISTORY_FRAMES, OBS_DIM), dtype=np.float32)
        self.count = 0

    @property
    def ready(self):
        return self.count >= HISTORY_FRAMES

    def reset(self):
        self._history.fill(0.0)
        self.count = 0

    def push(self, observation):
        frame = finite_vector(observation, OBS_DIM, "observation")
        if np.any(np.abs(frame) > OBS_CLIP):
            raise ValueError("observation must already be clipped to the training observation range")
        self._history[1:] = self._history[:-1].copy()
        self._history[0] = frame
        self.count += 1
        return self.get()

    def get(self):
        return self._history.reshape(1, HISTORY_DIM).copy()


def validate_rotation_matrix(rotation):
    """R_BS maps components expressed in sensor frame S to body frame B."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("sensor_to_body_rotation must be a finite 3x3 matrix")
    if (not np.allclose(matrix.T.dot(matrix), np.eye(3), atol=1e-6, rtol=0.0) or
            not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6, rtol=0.0)):
        raise ValueError("sensor_to_body_rotation must be a proper orthonormal rotation")
    return matrix.copy()


def rotation_rpy(rpy):
    """R_WS = Rz(yaw) Ry(pitch) Rx(roll), source frame to world.

    These are attitude angles in radians, not integrated Euler-angle rates.
    Their direction and mounting convention must be verified on the actual Go1.
    """
    angles = np.asarray(rpy, dtype=np.float64)
    if angles.shape != (3,) or not np.all(np.isfinite(angles)):
        raise ValueError("rpy must be a finite three-vector in radians")
    sr, sp, sy = np.sin(angles)
    cr, cp, cy = np.cos(angles)
    return np.asarray([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                       [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                       [-sp, cp * sr, cp * cr]], dtype=np.float64)


def rotation_quaternion_wxyz(quaternion):
    """Unitree wxyz quaternion -> R_WS; no hidden xyzw reinterpretation.

    Reject grossly invalid norms; normalize small sensor roundoff. The 5% norm
    tolerance is a deployment validity check, not a HIM training parameter.
    """
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("quaternion must be a finite wxyz four-vector")
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or abs(norm - 1.0) > 0.05:
        raise ValueError("quaternion norm must be within 0.05 of one")
    w, x, y, z = quat / norm
    return np.asarray([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                       [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                       [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def projected_gravity(attitude, method, sensor_to_body_rotation):
    """Return R_WB.T [0,0,-1], with R_WB = R_WS R_BS.T."""
    rotation_bs = validate_rotation_matrix(sensor_to_body_rotation)
    if method == "rpy":
        rotation_ws = rotation_rpy(attitude)
    elif method == "quaternion_wxyz":
        rotation_ws = rotation_quaternion_wxyz(attitude)
    else:
        raise ValueError("unsupported attitude source: {}".format(method))
    rotation_wb = rotation_ws.dot(rotation_bs.T)
    return rotation_wb.T.dot(np.asarray([0.0, 0.0, -1.0])).astype(np.float32)
