"""Simulator-independent episode metrics and baseline screening for Quiet V1.

All motion samples include start-up and terminal steps. Only the simulator's
touchdown measurement omits the first 0.5 seconds. A failed episode is never
replaced by the automatically reset robot's next episode.
"""
import math
import numpy as np

QUIET_EVALUATION_FORMAT_VERSION = 2


def _percentile(values, percentile=95):
    return float(np.percentile(values, percentile)) if values else None


def evaluation_conditions(suite="all"):
    flat = [
        {"name": "flat_forward_%.1f" % v, "terrain": "plane",
         "command": [v, 0.0, 0.0], "quiet_eligible": True}
        for v in (0.3, 0.6, 0.9)
    ]
    flat.extend([
        {"name": "flat_stand", "terrain": "plane", "command": [0., 0., 0.]},
        {"name": "flat_backward", "terrain": "plane", "command": [-0.3, 0., 0.]},
        {"name": "flat_side", "terrain": "plane", "command": [0., 0.3, 0.]},
        {"name": "flat_yaw", "terrain": "plane", "command": [0., 0., 0.5]},
    ])
    # choices refer to the original terrain generator's cumulative proportions:
    # [0.2 slopes, 0.2 rough slopes, 0.2 down stairs, 0.2 up stairs, 0.2 obstacles].
    rough = [
        {"name": "rough_" + name, "terrain": "trimesh", "choice": choice,
         "difficulty": 0.5, "command": [0.6, 0., 0.]}
        # Robots start at the central platform and walk outward. A positive
        # pyramid height therefore produces a descent, not an ascent.
        for name, choice in (("slope_down", 0.15), ("slope_up", 0.05),
                             ("rough_slope", 0.3), ("stairs_down", 0.7),
                             ("stairs_up", 0.5), ("obstacles", 0.9))
    ]
    push = [{"name": "flat_push", "terrain": "plane", "command": [0.6, 0., 0.],
             "push_time": 5.0, "push_velocity_y": 0.5}]
    suites = {"flat": flat, "rough": rough, "push": push, "all": flat + rough + push}
    if suite not in suites:
        raise ValueError("Unknown evaluation suite: " + suite)
    return suites[suite]


def condition_seed(name, split="validation"):
    """Stable seeds, independent of Python's randomized hash and suite order."""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    return (17000 if split == "validation" else 29000) + sum(
        (i + 1) * ord(char) for i, char in enumerate(name))


def validate_evaluation_schedule(conditions, duration, control_dt):
    """Validate the entire suite before allocating any simulator resources.

    A push is applied at the beginning of its control step. The final loop step
    starts at duration - control_dt, so a push at duration cannot be applied.
    """
    duration = float(duration)
    control_dt = float(control_dt)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Evaluation duration must be finite and positive")
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("Evaluation control period must be finite and positive")
    steps_float = duration / control_dt
    if not math.isfinite(steps_float):
        raise ValueError("Evaluation duration/control-period ratio is invalid")
    num_steps = int(round(steps_float))
    # Accept normal binary-float rounding, but not an extra partial control step.
    if num_steps < 1 or not math.isclose(num_steps * control_dt, duration,
                                        rel_tol=1e-12, abs_tol=control_dt * 1e-8):
        raise ValueError("Evaluation duration must contain whole control steps")
    push_steps = {}
    for condition in conditions:
        if "push_time" not in condition:
            continue
        push_time = float(condition["push_time"])
        if not math.isfinite(push_time) or push_time < 0:
            raise ValueError(condition["name"] + ": push time must be finite and nonnegative")
        push_float = push_time / control_dt
        if not math.isfinite(push_float):
            raise ValueError(condition["name"] + ": push schedule is invalid")
        push_step = int(round(push_float))
        if not math.isclose(push_step * control_dt, push_time,
                            rel_tol=1e-12, abs_tol=control_dt * 1e-8):
            raise ValueError(condition["name"] + ": push time must align with a control step")
        if push_step >= num_steps:
            raise ValueError(condition["name"] + ": duration must extend past the scheduled push")
        push_steps[condition["name"]] = push_step
    return num_steps, push_steps


class QuietEpisodeMetrics:
    """Aggregate pre-reset snapshots for one first episode per parallel robot."""
    def __init__(self, condition, episodes, dt, duration, foot_names,
                 torque_limits, action_scale=0.25, startup_seconds=0.5,
                 action_clip_limit=100.0, deployment_action_limit=1.0):
        self.condition = dict(condition)
        self.episodes = int(episodes)
        self.dt = float(dt)
        self.duration = float(duration)
        self.feet = list(foot_names)
        self.torque_limits = np.asarray(torque_limits)
        self.action_scale = float(action_scale)
        self.startup_seconds = float(startup_seconds)
        self.action_clip_limit = float(action_clip_limit)
        self.deployment_action_limit = float(deployment_action_limit)
        if any(not math.isfinite(value) or value <= 0 for value in
               (self.action_clip_limit, self.deployment_action_limit)):
            raise ValueError("Action metric limits must be finite and positive")
        n = self.episodes
        self.active = np.ones(n, dtype=bool)
        self.failed = np.zeros(n, dtype=bool)
        self.steps = np.zeros(n, dtype=np.int64)
        self.error_xy = np.zeros(n)
        self.error_yaw = np.zeros(n)
        self.velocity = np.zeros((n, 3))
        self.orientation = np.zeros(n)
        self.power = np.zeros(n)
        self.slip = np.zeros(n)
        self.contact_samples = np.zeros(n)
        self.action_clip = np.zeros(n)
        self.deployment_limit_exceeded = np.zeros(n)
        self.action_metric_steps = np.zeros(n, dtype=np.int64)
        self.torque_saturation = np.zeros(n)
        self.drift_squared = np.zeros(n)
        self.path = np.zeros(n)
        self.displacement_xy = np.zeros((n, 2))
        self.previous_position = None
        self.touch_count = np.zeros((n, len(self.feet)), dtype=np.int64)
        self.impact_count = np.zeros_like(self.touch_count)
        self.impact_invalid = np.zeros_like(self.touch_count)
        self.touch_samples = [[] for _ in self.feet]
        self.impact_peaks = [[] for _ in self.feet]
        self.impact_loads = [[] for _ in self.feet]
        self.event_records = []
        self.drift_values = []

    def set_initial_position(self, positions):
        self.previous_position = np.asarray(positions)[:, :2].copy()

    def add(self, snapshot, action_drift=None, raw_actions=None):
        """snapshot arrays are copied before any reset; each has N as first axis."""
        alive = self.active.copy()
        if not alive.any():
            return
        idx = np.flatnonzero(alive)
        self.steps[idx] += 1
        self.error_xy[idx] += np.asarray(snapshot["track_error"])[idx]
        self.error_yaw[idx] += np.asarray(snapshot["yaw_error"])[idx]
        self.velocity[idx] += np.asarray(snapshot["base_velocity"])[idx]
        gravity = np.asarray(snapshot["projected_gravity"])[idx, :2]
        self.orientation[idx] += np.sum(gravity * gravity, axis=-1)
        torque = np.asarray(snapshot["torques"])[idx]
        dof_vel = np.asarray(snapshot["dof_velocity"])[idx]
        self.power[idx] += np.sum(np.abs(torque * dof_vel), axis=-1)
        self.torque_saturation[idx] += np.mean(
            np.abs(torque) >= .99 * self.torque_limits, axis=-1)
        contacts = np.asarray(snapshot["contact"], dtype=bool)[idx]
        speed = np.linalg.norm(np.asarray(snapshot["foot_velocity"])[idx, :, :2], axis=-1)
        self.slip[idx] += np.sum(contacts & (speed > 0.2), axis=-1)
        self.contact_samples[idx] += np.sum(contacts, axis=-1)
        if raw_actions is not None:
            magnitude = np.abs(np.asarray(raw_actions)[idx])
            self.action_clip[idx] += np.mean(magnitude > self.action_clip_limit, axis=-1)
            self.deployment_limit_exceeded[idx] += np.mean(
                magnitude > self.deployment_action_limit, axis=-1)
            self.action_metric_steps[idx] += 1
        if action_drift is not None:
            delta_rad = np.asarray(action_drift)[idx] * self.action_scale
            self.drift_squared[idx] += np.mean(delta_rad * delta_rad, axis=-1)
            self.drift_values.extend(np.abs(delta_rad).ravel().tolist())
        position = np.asarray(snapshot["root_position"])[:, :2]
        if self.previous_position is not None:
            delta = position[idx] - self.previous_position[idx]
            self.path[idx] += np.linalg.norm(delta, axis=-1)
            self.displacement_xy[idx] += delta
        self.previous_position = position.copy()
        self.touch_count[idx] += np.asarray(snapshot["touchdown_count"])[idx]
        self.impact_count[idx] += np.asarray(snapshot["impact_count"])[idx]
        self.impact_invalid[idx] += np.asarray(snapshot["impact_invalid"])[idx]
        event = np.asarray(snapshot["touchdown_event"], dtype=bool)
        speeds = np.asarray(snapshot["touchdown_speed"])
        impacts = np.asarray(snapshot["impact_event"], dtype=bool)
        peaks = np.asarray(snapshot["impact_peak_samples"])
        loads = np.asarray(snapshot["impact_load_samples"])
        times = np.asarray(snapshot["episode_time"])
        for env in idx:
            for substep, foot in np.argwhere(event[env]):
                speed_value = float(speeds[env, substep, foot])
                self.touch_samples[foot].append(speed_value)
                sample_time = times[env] - self.dt + (substep + 1) * self.dt / event.shape[1]
                self.event_records.append({"episode": int(env), "time": float(sample_time),
                                           "substep": int(substep), "foot": self.feet[foot],
                                           "touchdown_speed": speed_value})
            for substep, foot in np.argwhere(impacts[env]):
                self.impact_peaks[foot].append(float(peaks[env, substep, foot]))
                self.impact_loads[foot].append(float(loads[env, substep, foot]))
        failed = np.asarray(snapshot["failed"], dtype=bool)
        self.failed |= alive & failed
        done = np.asarray(snapshot["done"], dtype=bool)
        self.active &= ~done

    def finish(self):
        count = int(self.steps.sum())
        if count == 0:
            raise ValueError("Cannot summarize an evaluation without samples")
        action_count = int(self.action_metric_steps.sum())
        touch = [value for foot in self.touch_samples for value in foot]
        peaks = [value for foot in self.impact_peaks for value in foot]
        loads = [value for foot in self.impact_loads for value in foot]
        elapsed = self.steps * self.dt
        quiet_time = float(np.maximum(0., elapsed - self.startup_seconds).sum())
        mean_vel = self.velocity.sum(axis=0) / count
        command = np.asarray(self.condition["command"])
        command_norm = float(np.linalg.norm(command[:2]))
        projected_speed = (float(np.dot(mean_vel[:2], command[:2]) / command_norm)
                           if command_norm > 1e-8 else None)
        traveled = (self.displacement_xy.dot(command[:2] / command_norm)
                    if command_norm > 1e-8 else np.zeros(self.episodes))
        horizon = elapsed >= self.duration - .5 * self.dt
        completed = (~self.failed) & horizon
        if command_norm > 1e-8:
            completed &= traveled >= .8 * command_norm * self.duration
        squared_sum = float(np.square(touch).sum())
        path = float(self.path.sum())
        episode_records = []
        for i in range(self.episodes):
            denominator = max(1, int(self.steps[i]))
            episode_records.append({
                "episode": i, "failed": bool(self.failed[i]),
                "completed": bool(completed[i]), "duration_s": float(elapsed[i]),
                "path_m": float(self.path[i]), "progress_m": float(traveled[i]),
                "tracking_xy_rmse": math.sqrt(float(self.error_xy[i]) / denominator),
                "tracking_yaw_rmse": math.sqrt(float(self.error_yaw[i]) / denominator),
                "touchdown_counts": self.touch_count[i].tolist(),
            })
        return {
            "name": self.condition["name"], "condition": self.condition,
            "episodes": self.episodes, "failure_count": int(self.failed.sum()),
            "failure_rate": float(self.failed.mean()), "completion_rate": float(completed.mean()),
            "sample_count": count, "elapsed_s": float(elapsed.sum()), "quiet_elapsed_s": quiet_time,
            "tracking_xy_rmse": math.sqrt(float(self.error_xy.sum()) / count),
            "tracking_yaw_rmse": math.sqrt(float(self.error_yaw.sum()) / count),
            "mean_base_velocity": mean_vel.tolist(), "actual_forward_speed": float(mean_vel[0]),
            "actual_command_speed": projected_speed,
            "orientation_rms": math.sqrt(float(self.orientation.sum()) / count),
            "mechanical_power_mean": float(self.power.sum() / count),
            "slip_contact_fraction": float(self.slip.sum() / max(1., self.contact_samples.sum())),
            "action_clip_fraction": float(self.action_clip.sum() / action_count) if action_count else None,
            "action_clip_limit": self.action_clip_limit,
            "deployment_action_limit_exceeded_fraction": (
                float(self.deployment_limit_exceeded.sum() / action_count) if action_count else None),
            "deployment_action_limit": self.deployment_action_limit,
            "action_metric_sample_count": action_count,
            "torque_saturation_fraction": float(self.torque_saturation.sum() / count),
            "action_drift_rad_rms": math.sqrt(float(self.drift_squared.sum()) / count),
            "action_drift_rad_p95": _percentile(self.drift_values),
            "action_drift_rad_max": max(self.drift_values) if self.drift_values else None,
            "touchdown_valid": bool(touch), "touchdown_count": len(touch),
            "touchdown_velocity_rms": math.sqrt(squared_sum / len(touch)) if touch else None,
            "touchdown_velocity_p95": _percentile(touch),
            "touchdown_cost_per_second": squared_sum / quiet_time if quiet_time > 0 else None,
            "touchdown_cost_per_meter": squared_sum / path if path > 1e-8 else None,
            "contact_window_peak_p95": _percentile(peaks),
            "contact_window_load_p95": _percentile(loads),
            "impact_complete_count": len(peaks), "impact_invalid_count": int(self.impact_invalid.sum()),
            "per_foot": {foot: {"touchdown_count": int(self.touch_count[:, i].sum()),
                                "impact_count": int(self.impact_count[:, i].sum()),
                                "touchdown_speeds": self.touch_samples[i],
                                "impact_peaks": self.impact_peaks[i],
                                "impact_loads": self.impact_loads[i]}
                         for i, foot in enumerate(self.feet)},
            "episode_records": episode_records, "touchdown_events": self.event_records,
        }


def performance_screen(candidate, baseline):
    """Per-bucket V1 engineering screen. It is not a statistical guarantee."""
    reasons = []
    pairs = (("tracking_xy_rmse", .02), ("tracking_yaw_rmse", .03))
    for field, floor in pairs:
        limit = baseline[field] + max(.05 * baseline[field], floor)
        if not math.isfinite(candidate[field]) or candidate[field] > limit:
            reasons.append(field + " exceeds motion budget")
    command_norm = np.linalg.norm(candidate["condition"]["command"][:2])
    if command_norm > 1e-8:
        speed = candidate["actual_command_speed"]
        if speed is None or not math.isfinite(speed) or speed < .95 * baseline["actual_command_speed"]:
            reasons.append("actual command-direction speed below 95% of baseline")
    if candidate["failure_count"] > baseline["failure_count"]:
        reasons.append("failure count exceeds baseline")
    if candidate["slip_contact_fraction"] > 1.05 * baseline["slip_contact_fraction"]:
        reasons.append("slip fraction exceeds motion budget")
    base_peak = baseline["contact_window_peak_p95"]
    candidate_peak = candidate["contact_window_peak_p95"]
    if base_peak is not None and (candidate_peak is None or candidate_peak > 1.05 * base_peak):
        reasons.append("contact-window force p95 exceeds motion budget")
    return {"passed": not reasons, "reasons": reasons}


def select_candidate(candidate_report, baseline_report, require_quiet=True):
    """Screen every bucket, then require >=10% quieter at each forward speed."""
    candidate = {item["name"]: item for item in candidate_report["conditions"]}
    baseline = {item["name"]: item for item in baseline_report["conditions"]}
    reasons = []
    details = {}
    if set(candidate) != set(baseline):
        return {"accepted": False, "reasons": ["evaluation conditions differ"], "score": None}
    if candidate_report.get("split") != baseline_report.get("split"):
        return {"accepted": False, "reasons": ["evaluation splits differ"], "score": None}
    for field in ("format_version", "duration_s", "seed_offset"):
        if candidate_report.get(field) != baseline_report.get(field):
            return {"accepted": False, "reasons": [field + " differs"], "score": None}
    if any(candidate[name]["condition"] != baseline[name]["condition"]
           or candidate[name].get("seed") != baseline[name].get("seed") for name in candidate):
        return {"accepted": False, "reasons": ["condition settings or paired seeds differ"], "score": None}
    if any(candidate[name]["episodes"] != baseline[name]["episodes"] for name in candidate):
        return {"accepted": False, "reasons": ["episode counts differ"], "score": None}
    for name in candidate:
        for field in ("physical_dt_s", "control_dt_s", "action_clip_limit", "deployment_action_limit"):
            # Legacy synthetic reports may omit these fields. A paired report
            # that contains them on either side must contain equal values.
            if candidate[name].get(field) != baseline[name].get(field):
                return {"accepted": False, "reasons": [name + ": " + field + " differs"],
                        "score": None}
    scores = []
    eligible_count = 0
    for name, item in candidate.items():
        screen = performance_screen(item, baseline[name])
        local = list(screen["reasons"])
        if "push_time" in item["condition"]:
            if not item.get("push_applied", False) or not baseline[name].get("push_applied", False):
                local.append("scheduled push was not applied; push validation is incomplete")
        if item["condition"].get("terrain") != "plane":
            candidate_exits = item.get("terrain_platform_exit_count")
            baseline_exits = baseline[name].get("terrain_platform_exit_count")
            # Older reports may omit exposure metadata. Current reports always
            # provide it, and cannot claim terrain preservation on a flat spawn.
            if candidate_exits is not None or baseline_exits is not None:
                if not candidate_exits or not baseline_exits:
                    local.append("central terrain platform was not traversed; rough validation is incomplete")
        if item["condition"].get("quiet_eligible", False):
            eligible_count += 1
            if not item["touchdown_valid"] or not baseline[name]["touchdown_valid"]:
                if require_quiet:
                    local.append("touchdown samples missing; quietness is invalid")
            else:
                for field in ("touchdown_velocity_rms", "touchdown_velocity_p95"):
                    base_value = baseline[name][field]
                    value = item[field]
                    if not require_quiet:
                        continue
                    if base_value is None or not math.isfinite(base_value) or base_value <= 0:
                        local.append(field + " baseline has no positive quietness cost")
                    elif value is None or not math.isfinite(value):
                        local.append(field + " candidate measurement is invalid")
                    elif require_quiet and value > .9 * base_value:
                        local.append(field + " did not improve by 10%")
                scores.append(item["touchdown_cost_per_second"])
        details[name] = {"passed": not local, "reasons": local}
        reasons.extend(name + ": " + reason for reason in local)
    if require_quiet and not eligible_count:
        reasons.append("no forward-flat quietness conditions")
    return {"accepted": not reasons, "reasons": reasons, "per_condition": details,
            "score": float(np.mean(scores)) if scores else None,
            "criteria": {"tracking_relative": .05, "xy_rmse_floor_mps": .02,
                         "yaw_rmse_floor_radps": .03, "speed_fraction": .95,
                         "slip_relative": .05, "force_relative": .05,
                         "quiet_improvement_fraction": .10}}


def evaluate_policy(*args, **kwargs):
    """Lazy simulator wrapper; pure metric tests never import Isaac Gym."""
    from legged_gym.scripts.evaluate_quiet import evaluate_policy as implementation
    return implementation(*args, **kwargs)
