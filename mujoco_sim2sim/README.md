# Go1 HIM `policy.pt` MuJoCo Sim2Sim

This module splits the sim2sim implementation into parts:

- `policy_runner.py`: TorchScript policy inference wrapper.
- `config.py`: Go1 policy/control/obs settings.
- `mujoco_env.py`: MuJoCo model/data init, observation conversion, PD control stepping.
- `keyboard.py`: runtime command control (`W/S/A/D/Q/E/SPACE`).
- `main.py`: executable loop integrating all components.

## Run

```bash
python -m projects.himloco.mujoco_sim2sim.main \
  --policy /absolute/path/to/policy.pt \
  --xml legged_gym/resources/robots/go1/xml/go1.xml \
  --steps 4000
```

```bash
python -m mujoco_sim2sim.main   --policy legged_gym/logs/rough_go1/exported/policies/policy.pt   --xml legged_gym/resources/robots/go1/xml/go1.xml   --steps 4000 --debug --print-every 100   --cmd-x 0.5 --heading-command --cmd-heading 0.0
```

Headless mode:

```bash
python -m projects.himloco.mujoco_sim2sim.main \
  --policy /absolute/path/to/policy.pt \
  --headless
```

## Observation mapping (45-d one-step)

The code reproduces HIM actor one-step inputs:

1. commands `(vx, vy, yaw)`
2. base angular velocity in body frame (scaled)
3. projected gravity in body frame
4. `(dof_pos - default_dof_pos)` (scaled)
5. `dof_vel` (scaled)
6. `last_actions`

and then stacks 6 history frames into 270-d `obs_history`.

## Notes

- This module is Go1-only by design.
- Keep joint order and actuator mapping aligned with Go1 training config.
- MuJoCo XML/mesh consistency matters; if loading fails, validate XML resources first.

## MuJoCo documentation targets to verify during deployment

- API initialization: `MjModel.from_xml_path`, `MjData`, `mj_step`, `mj_resetData`.
- XML `<option>` parameters: timestep, gravity, integrator.
- Quaternion convention (wxyz) and frame transforms.
- Control path via `data.ctrl` / actuators.

(These are the key official topics to cross-check against your installed MuJoCo version.)