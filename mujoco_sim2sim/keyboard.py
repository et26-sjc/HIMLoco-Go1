from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommandState:
    vx: float
    vy: float
    wyaw: float
    heading: float = 0.0


class KeyboardCommander:
    """Simple key mapping for MuJoCo viewer callback.

    Controls:
      W/S: +/- vx
      A/D: +/- vy
      Q/E: +/- yaw rate
      SPACE: zero all commands
    """

    def __init__(self, vx: float, vy: float, wyaw: float, heading: float = 0.0, step: float = 0.1):
        self.state = CommandState(vx=vx, vy=vy, wyaw=wyaw, heading=heading)
        self.step = step

    def on_key(self, keycode: int) -> None:
        # GLFW keycodes used by mujoco.viewer
        if keycode in (87,):  # W
            self.state.vx += self.step
        elif keycode in (83,):  # S
            self.state.vx -= self.step
        elif keycode in (65,):  # A
            self.state.vy += self.step
        elif keycode in (68,):  # D
            self.state.vy -= self.step
        elif keycode in (81,):  # Q
            self.state.heading += self.step
        elif keycode in (69,):  # E
            self.state.heading -= self.step
        elif keycode in (32,):  # SPACE
            self.state.vx = 0.0
            self.state.vy = 0.0
            self.state.wyaw = 0.0