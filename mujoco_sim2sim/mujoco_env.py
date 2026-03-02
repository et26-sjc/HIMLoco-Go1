from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MujocoSimConfig


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def rotate_vec_world_to_body_wxyz(q_body_world: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    # v_body = q^-1 * v * q
    vq = np.array([0.0, v_world[0], v_world[1], v_world[2]], dtype=np.float64)
    q_inv = quat_conjugate_wxyz(q_body_world)
    out = quat_mul_wxyz(quat_mul_wxyz(q_inv, vq), q_body_world)
    return out[1:]


def wrap_to_pi(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def quat_to_yaw_wxyz(q: np.ndarray) -> float:
    """Yaw (around +z) from MuJoCo quaternion wxyz."""
    w, x, y, z = q
    # standard yaw from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


@dataclass
class MujocoGo1Env:
    cfg: MujocoSimConfig

    def __post_init__(self) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.model_xml))
        self.data = mujoco.MjData(self.model)

                # ---- resolve policy cfg (STRICT, no fallback) ----
        # print cfg fields once for debugging if something is wrong
        if self.cfg.debug:
            try:
                print("[mujoco_env] cfg fields:", sorted(vars(self.cfg).keys()))
            except Exception:
                print("[mujoco_env] cfg fields: <unavailable>")

        if hasattr(self.cfg, "policy") and (self.cfg.policy is not None):
            self.pcfg = self.cfg.policy
        elif hasattr(self.cfg, "go1") and (self.cfg.go1 is not None):
            self.pcfg = self.cfg.go1
        else:
            raise AttributeError(
                "Cannot resolve policy config. Expected cfg.policy or cfg.go1 to exist.\n"
                f"cfg type: {type(self.cfg)}\n"
                f"cfg fields: {sorted(vars(self.cfg).keys()) if hasattr(self.cfg, '__dict__') else '<no __dict__>'}"
            )

        # minimal sanity checks (fail fast, no silent mismatch)
        for k in ("sim_dt", "decimation", "num_actions", "num_one_step_obs", "history_len",
                  "action_scale", "hip_reduction", "kp", "kd",
                  "obs_scale_ang_vel", "obs_scale_dof_pos", "obs_scale_dof_vel",
                  "default_dof_pos", "dof_names"):
            if not hasattr(self.pcfg, k):
                raise AttributeError(f"policy cfg missing required field: {k}")

        # align dt
        self.model.opt.timestep = float(self.pcfg.sim_dt)

        self.n_act = self.pcfg.num_actions
        self.default_dof_pos = np.array(self.pcfg.default_dof_pos, dtype=np.float64)
        self.last_actions = np.zeros(self.n_act, dtype=np.float64)

        self.obs_history = np.zeros(
            self.pcfg.num_one_step_obs * self.pcfg.history_len,
            dtype=np.float32,
        )

        # raw command storage: (vx, vy, yaw_rate) where yaw_rate may be overwritten by heading logic
        self.cmd_raw = np.array([self.cfg.cmd_x, self.cfg.cmd_y, self.cfg.cmd_yaw], dtype=np.float64)
        self.cmd_heading = float(self.cfg.cmd_heading)

        # debug caches
        self._dbg_enabled = bool(self.cfg.debug)
        self._dbg = {
            "base_lin_vel_body": np.zeros(3, dtype=np.float64),
            "base_ang_vel_body": np.zeros(3, dtype=np.float64),
            "projected_gravity": np.zeros(3, dtype=np.float64),
            "tau": np.zeros(self.n_act, dtype=np.float64),
            "tau_sat_frac": 0.0,      # fraction joints saturating torque
            "tau_max_ratio": 0.0,     # max(|tau|/limit)
            "action_sat_frac": 0.0,   # fraction actions near ±1
            "cmd_obs": np.zeros(3, dtype=np.float64),
            "q_err_max": 0.0,
            "hist_block_mean": np.zeros(self.pcfg.history_len, dtype=np.float64),
            "hist_block_std": np.zeros(self.pcfg.history_len, dtype=np.float64),
            "has_torque_limit": False,
        }

        self._setup_indices()
        self.reset()

    def _setup_indices(self) -> None:
        m = self.model
        mujoco = self.mujoco

        self.body_trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")

        # joint order must match policy action order
        self.joint_ids: list[int] = []
        self.qpos_ids: list[int] = []
        self.qvel_ids: list[int] = []
        self.actuator_ids: list[int] = []

        for joint_name in self.pcfg.dof_names:
            j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if j_id < 0:
                raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")
            self.joint_ids.append(int(j_id))
            self.qpos_ids.append(int(m.jnt_qposadr[j_id]))
            self.qvel_ids.append(int(m.jnt_dofadr[j_id]))

            act_name = joint_name.replace("_joint", "")
            a_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if a_id < 0:
                raise ValueError(f"Actuator not found in MuJoCo model: {act_name}")
            self.actuator_ids.append(int(a_id))

    def reset(self) -> np.ndarray:
        mujoco = self.mujoco
        mujoco.mj_resetData(self.model, self.data)

        # set base pose close to Go1 init state
        self.data.qpos[0:3] = np.array([0.0, 0.0, 0.42], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.data.qpos[np.array(self.qpos_ids, dtype=np.int32)] = self.default_dof_pos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.last_actions[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.obs_history[:] = 0.0
        one = self._build_one_step_obs()
        for _ in range(self.pcfg.history_len):
            self._push_obs(one)

        if self._dbg_enabled:
            self._update_debug_kinematics()
            self._dbg["tau"][:] = 0.0
            self._dbg["tau_sat_frac"] = 0.0
            self._dbg["tau_max_ratio"] = 0.0
            self._dbg["action_sat_frac"] = 0.0

        return self.obs_history.copy()

    def set_command(self, vx: float, vy: float, wyaw: float, heading: float | None = None) -> None:
        self.cmd_raw[:] = [vx, vy, wyaw]
        if heading is not None:
            self.cmd_heading = float(heading)

    def _get_base_quat_wxyz(self) -> np.ndarray:
        # freejoint quaternion is qpos[3:7], wxyz (MuJoCo convention)
        return self.data.qpos[3:7].copy()

    def _update_debug_kinematics(self) -> None:
        q_body_world = self._get_base_quat_wxyz()

        # freejoint linear vel (world) is qvel[0:3], angular vel (world) is qvel[3:6]
        base_lin_vel_world = self.data.qvel[0:3].copy()
        base_ang_vel_world = self.data.qvel[3:6].copy()

        self._dbg["base_lin_vel_body"] = rotate_vec_world_to_body_wxyz(q_body_world, base_lin_vel_world)
        self._dbg["base_ang_vel_body"] = rotate_vec_world_to_body_wxyz(q_body_world, base_ang_vel_world)

        gravity_world = np.array(self.cfg.gravity_vec_world, dtype=np.float64)
        self._dbg["projected_gravity"] = rotate_vec_world_to_body_wxyz(q_body_world, gravity_world)

    def _build_one_step_obs(self) -> np.ndarray:
        q = self.data.qpos[np.array(self.qpos_ids, dtype=np.int32)]
        dq = self.data.qvel[np.array(self.qvel_ids, dtype=np.int32)]

        q_body_world = self._get_base_quat_wxyz()

        # free-joint angular velocity slice
        base_ang_vel_world = self.data.qvel[3:6].copy()
        base_ang_vel_body = rotate_vec_world_to_body_wxyz(q_body_world, base_ang_vel_world)

        gravity_world = np.array(self.cfg.gravity_vec_world, dtype=np.float64)
        projected_gravity = rotate_vec_world_to_body_wxyz(q_body_world, gravity_world)

        # --- command processing (match LeggedGym) ---
        cmd = self.cmd_raw.copy()
        if self.cfg.heading_command:
            # compute yaw-rate from heading error (match training distribution)
            yaw = quat_to_yaw_wxyz(q_body_world)
            err = wrap_to_pi(self.cmd_heading - yaw)
            cmd[2] = float(np.clip(0.5 * err, -2.0, 2.0))
            if self._dbg_enabled:
                self._dbg["yaw"] = yaw
                self._dbg["cmd_yaw_rate"] = cmd[2]

        cmd_scale = np.array(getattr(self.pcfg, "commands_scale", (2.0, 2.0, 0.25)), dtype=np.float64)
        cmd_obs = (cmd * cmd_scale).astype(np.float64)

        if self._dbg_enabled:
            self._dbg["cmd_obs"] = cmd_obs.copy()
            self._dbg["q_err_max"] = float(np.max(np.abs(q - self.default_dof_pos)))

        one = np.concatenate(
            [
                cmd_obs,
                base_ang_vel_body * self.pcfg.obs_scale_ang_vel,
                projected_gravity,
                (q - self.default_dof_pos) * self.pcfg.obs_scale_dof_pos,
                dq * self.pcfg.obs_scale_dof_vel,
                self.last_actions,
            ]
        ).astype(np.float32)

        if one.shape[0] != self.pcfg.num_one_step_obs:
            raise ValueError(
                f"one_step_obs dim mismatch: expected {self.pcfg.num_one_step_obs}, got {one.shape[0]}"
            )
        return one

    def _push_obs(self, one_step_obs: np.ndarray) -> None:
        n = self.pcfg.num_one_step_obs
        self.obs_history[n:] = self.obs_history[:-n]
        self.obs_history[:n] = one_step_obs

    def _compute_torque(self, action: np.ndarray) -> np.ndarray:
        q = self.data.qpos[np.array(self.qpos_ids, dtype=np.int32)]
        dq = self.data.qvel[np.array(self.qvel_ids, dtype=np.int32)]

        a = np.clip(action.astype(np.float64), -1.0, 1.0)
        a_scaled = a * self.pcfg.action_scale
        # hip reduction on FL/FR/RL/RR hip indices: 0,3,6,9
        a_scaled[[0, 3, 6, 9]] *= self.pcfg.hip_reduction

        q_target = self.default_dof_pos + a_scaled
        tau = self.pcfg.kp * (q_target - q) - self.pcfg.kd * dq

        if self._dbg_enabled:
            self._dbg["tau"] = tau.copy()

        return tau

    def _update_debug_saturation(self, action: np.ndarray, tau: np.ndarray) -> None:
        # action saturation
        a = np.clip(action.astype(np.float64), -1.0, 1.0)
        self._dbg["action_sat_frac"] = float(np.mean(np.abs(a) > 0.99))

        # torque saturation (relative to actuator forcerange if set)
        act_ids = np.array(self.actuator_ids, dtype=np.int32)
        fr = self.model.actuator_forcerange[act_ids]  # (nu, 2), may be zeros if not set

        limit = np.maximum(np.abs(fr[:, 0]), np.abs(fr[:, 1]))
        has_limit = bool(np.any(limit > 1e-6))
        self._dbg["has_torque_limit"] = has_limit
        # if not set, treat as no limit
        limit = np.where(limit < 1e-9, np.inf, limit)

        ratio = np.abs(tau) / limit
        self._dbg["tau_max_ratio"] = float(np.max(ratio))
        self._dbg["tau_sat_frac"] = float(np.mean(ratio > 0.98))

    def step(self, action: np.ndarray) -> np.ndarray:
        mujoco = self.mujoco

        tau = self._compute_torque(action)
        self.last_actions[:] = np.clip(action, -1.0, 1.0)

        if self._dbg_enabled:
            self._update_debug_saturation(action, tau)

        for _ in range(self.pcfg.decimation):
            self.data.ctrl[np.array(self.actuator_ids, dtype=np.int32)] = tau
            mujoco.mj_step(self.model, self.data)

        if self._dbg_enabled:
            self._update_debug_kinematics()

        one = self._build_one_step_obs()
        self._push_obs(one)
        if self._dbg_enabled:
            n = self.pcfg.num_one_step_obs
            H = self.pcfg.history_len
            means = []
            stds = []
            for i in range(H):
                blk = self.obs_history[i*n:(i+1)*n]
                means.append(float(np.mean(blk)))
                stds.append(float(np.std(blk)))
            self._dbg["hist_block_mean"] = np.array(means, dtype=np.float64)
            self._dbg["hist_block_std"] = np.array(stds, dtype=np.float64)
        return self.obs_history.copy()

    def get_debug(self) -> dict:
        """Return debug stats (empty dict if debug disabled)."""
        if not self._dbg_enabled:
            return {}

        out = dict(self._dbg)
        # copy arrays to avoid mutation from caller
        out["base_lin_vel_body"] = out["base_lin_vel_body"].copy()
        out["base_ang_vel_body"] = out["base_ang_vel_body"].copy()
        out["projected_gravity"] = out["projected_gravity"].copy()
        out["tau"] = out["tau"].copy()
        return out
