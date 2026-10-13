"""Proposed action a_t (Eq. 1).

An action carries both the semantic, skill-level description used by the
semantic monitors (task tag, source/destination, volumes, temperatures, ...)
and the low-level control command checked by the kinematic ActionFilter
(end-effector target, joint position target, joint velocity).  A policy can
fill either level; the LabUtopia bridge fills both.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

SEMANTIC_PARAMS: tuple[str, ...] = (
    "volume", "speed", "pour_rate", "target_temperature", "grip_force", "tilt_angle", "duration",
)


@dataclass
class Action:
    tag: str
    src: str | None = None  # container/object being manipulated (poured from, heated, ...)
    dst: str | None = None  # receiving container, equipment id, or zone name
    volume: float | None = None  # mL transferred
    speed: float | None = None  # end-effector speed (m/s)
    pour_rate: float | None = None  # mL/s
    target_temperature: float | None = None  # degC
    grip_force: float | None = None  # N
    tilt_angle: float | None = None  # deg
    duration: float | None = None  # s
    target_pos: np.ndarray | None = None  # end-effector target (robot base frame)
    q_target: np.ndarray | None = None  # joint position target (7,)
    qdot: np.ndarray | None = None  # joint velocity command (7,)
    gripper: float | None = None  # gripper opening command (m)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("target_pos", "q_target", "qdot"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, np.asarray(v, dtype=float))

    def copy(self) -> "Action":
        return copy.deepcopy(self)

    def params(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in SEMANTIC_PARAMS if getattr(self, k) is not None}

    def describe(self) -> str:
        parts = [self.tag]
        if self.src:
            parts.append(f"src={self.src}")
        if self.dst:
            parts.append(f"dst={self.dst}")
        for k, v in self.params().items():
            parts.append(f"{k}={v:g}")
        if self.target_pos is not None:
            parts.append("target=" + np.array2string(self.target_pos, precision=3))
        return " ".join(parts)

    def diff(self, other: "Action") -> dict[str, tuple[Any, Any]]:
        out = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                same = (a is None and b is None) or (a is not None and b is not None and np.allclose(a, b))
            else:
                same = a == b
            if not same:
                out[f.name] = (a, b)
        return out
