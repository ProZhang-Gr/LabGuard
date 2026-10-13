"""Franka Emika Panda kinematics and the kinematic ActionFilter (Sec. 3.5,
Eq. 8-11).

FK uses the manufacturer's modified (Craig) DH parameters.  The filter
operates on the low-level control command of an action:

  Eq. 8   workspace clamp        p_target <- clip(p_target, p_min + m, p_max - m)
  Eq. 9   velocity clamp         qdot_i   <- clip(qdot_i, -a*qdot_max_i, a*qdot_max_i)
  Eq. 10  joint-position barrier q_target <- clip(q_target, q_min + d, q_max - d)
  Eq. 11  joint-velocity barrier (q_min + d - q)/dt <= qdot <= (q_max - d - q)/dt
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from labguard.core.action import Action
from labguard.core.franka import (  # noqa: F401  (re-exported)
    FLANGE_D, GRIPPER_D, PANDA_DH, Q_HOME, Q_MAX, Q_MIN, QDOT_MAX, PandaKinematics,
)
from labguard.core.state import LabState


@dataclass
class FilterEvent:
    kind: str  # workspace_clamp | velocity_clamp | joint_position_barrier | joint_velocity_barrier
    detail: str
    magnitude: float = 0.0


@dataclass
class ActionFilterConfig:
    alpha: float = 0.8  # velocity scaling factor (Eq. 9)
    delta: float = 0.05  # joint barrier margin (rad, Eq. 10-11)
    workspace_margin: float = 0.01  # m (Eq. 8)
    dt: float = 0.05
    workspace_zone: str = "workspace"
    enable_workspace: bool = True
    enable_velocity: bool = True
    enable_joint_barrier: bool = True


@dataclass
class ActionFilter:
    """Kinematic safety filter applied after the semantic checks."""

    config: ActionFilterConfig = field(default_factory=ActionFilterConfig)
    kin: PandaKinematics = field(default_factory=PandaKinematics)

    def workspace_bounds(self, state: LabState) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = state.zones[self.config.workspace_zone]
        m = self.config.workspace_margin
        return np.asarray(lo, dtype=float) + m, np.asarray(hi, dtype=float) - m

    def __call__(self, state: LabState, action: Action) -> tuple[Action, list[FilterEvent]]:
        cfg = self.config
        out = action.copy()
        events: list[FilterEvent] = []
        q = state.robot.q

        # Eq. 8: Cartesian workspace clamp of the end-effector target
        if cfg.enable_workspace and out.target_pos is not None and cfg.workspace_zone in state.zones:
            lo, hi = self.workspace_bounds(state)
            clipped = np.clip(out.target_pos, lo, hi)
            if not np.allclose(clipped, out.target_pos):
                d = float(np.linalg.norm(clipped - out.target_pos))
                events.append(FilterEvent("workspace_clamp", f"target moved {d:.3f} m into workspace", d))
                out.target_pos = clipped
                if out.q_target is not None:  # keep the joint target consistent with the new target
                    out.q_target, _ = self.kin.ik_position(clipped, out.q_target)

        # Eq. 10: joint-position barrier on q_target
        if cfg.enable_joint_barrier and out.q_target is not None:
            lo, hi = Q_MIN + cfg.delta, Q_MAX - cfg.delta
            clipped = np.clip(out.q_target, lo, hi)
            if not np.allclose(clipped, out.q_target):
                d = float(np.max(np.abs(clipped - out.q_target)))
                events.append(FilterEvent("joint_position_barrier", f"q_target clipped by {d:.3f} rad", d))
                out.q_target = clipped

        # Commanded joint velocity: explicit qdot, or implied by q_target over one control step
        qdot = out.qdot
        if qdot is None and out.q_target is not None:
            qdot = (out.q_target - q) / cfg.dt
        if qdot is not None:
            new_qdot = qdot.copy()
            # Eq. 9: velocity clamp
            if cfg.enable_velocity:
                vmax = cfg.alpha * QDOT_MAX
                clamped = np.clip(new_qdot, -vmax, vmax)
                if not np.allclose(clamped, new_qdot):
                    ratio = float(np.max(np.abs(new_qdot) / vmax))
                    events.append(FilterEvent("velocity_clamp", f"joint speed {ratio:.2f}x alpha*limit", ratio))
                    new_qdot = clamped
            # Eq. 11: discrete-time joint-velocity barrier
            if cfg.enable_joint_barrier:
                lo = (Q_MIN + cfg.delta - q) / cfg.dt
                hi = (Q_MAX - cfg.delta - q) / cfg.dt
                barred = np.clip(new_qdot, np.minimum(lo, 0.0), np.maximum(hi, 0.0))
                if not np.allclose(barred, new_qdot):
                    events.append(FilterEvent("joint_velocity_barrier", "joint velocity limited near q-limit",
                                              float(np.max(np.abs(barred - new_qdot)))))
                    new_qdot = barred
            if not np.allclose(new_qdot, qdot):
                if out.qdot is not None:
                    out.qdot = new_qdot
                if out.q_target is not None:
                    out.q_target = q + new_qdot * cfg.dt
        return out, events

    def violations(self, state: LabState, action: Action) -> list[FilterEvent]:
        """Events the filter *would* produce (used for offline VR re-scoring)."""
        return self(state, action)[1]
