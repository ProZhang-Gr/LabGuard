"""Lab state s_t observed by the safety layer (Sec. 3.1, 3.4).

The state is deliberately simulator-agnostic: the toy simulator writes it
directly, and the LabUtopia adapter maintains it as a shadow state that is
synchronised with Isaac Sim object poses every control step.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from labguard.ir.registry import DEFAULT_ZONES

Box = tuple[tuple[float, float, float], tuple[float, float, float]]
EPS_VOLUME = 1e-6


def _vec(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3)


@dataclass
class Container:
    id: str
    kind: str = "beaker"  # beaker | flask | bottle | test_tube | cylinder | waste_container
    contents: dict[str, float] = field(default_factory=dict)  # substance id -> mL
    capacity: float = 250.0  # mL
    temperature: float = 22.0  # degC
    lid_open: bool = True
    has_lid: bool = True
    open_time: float = 0.0  # s since the lid was opened
    label: str | None = None
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    intact: bool = True
    clean: bool = True
    dry: bool = True
    upright: bool = True

    def __post_init__(self) -> None:
        self.position = _vec(self.position)

    @property
    def volume(self) -> float:
        return float(sum(self.contents.values()))

    @property
    def fill_ratio(self) -> float:
        return self.volume / self.capacity if self.capacity > 0 else 0.0

    @property
    def substances(self) -> set[str]:
        return {s for s, v in self.contents.items() if v > EPS_VOLUME}

    def add(self, mixture: dict[str, float]) -> None:
        for s, v in mixture.items():
            if v > 0:
                self.contents[s] = self.contents.get(s, 0.0) + v

    def remove(self, volume: float) -> dict[str, float]:
        """Remove `volume` mL with the current composition; returns the removed mixture."""
        total = self.volume
        if total <= EPS_VOLUME or volume <= 0:
            return {}
        frac = min(1.0, volume / total)
        out = {s: v * frac for s, v in self.contents.items()}
        for s in list(self.contents):
            self.contents[s] -= out[s]
            if self.contents[s] <= EPS_VOLUME:
                del self.contents[s]
        return out


@dataclass
class Equipment:
    id: str
    kind: str  # registry entity id: hot_plate | fume_hood | centrifuge
    flags: dict[str, bool] = field(default_factory=dict)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    setpoint: float | None = None  # e.g. hot-plate temperature

    def __post_init__(self) -> None:
        self.position = _vec(self.position)


@dataclass
class RobotState:
    q: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]))
    qdot: np.ndarray = field(default_factory=lambda: np.zeros(7))
    ee_pos: np.ndarray = field(default_factory=lambda: np.array([0.307, 0.0, 0.487]))
    holding: str | None = None
    gripper_force: float = 0.0
    speed: float = 0.0  # Cartesian end-effector speed (m/s)

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float)
        self.qdot = np.asarray(self.qdot, dtype=float)
        self.ee_pos = _vec(self.ee_pos)


@dataclass
class LabState:
    containers: dict[str, Container] = field(default_factory=dict)
    equipment: dict[str, Equipment] = field(default_factory=dict)
    robot: RobotState = field(default_factory=RobotState)
    ppe: set[str] = field(default_factory=set)
    done_steps: list[str] = field(default_factory=list)
    zones: dict[str, Box] = field(default_factory=lambda: dict(DEFAULT_ZONES))
    ambient_temperature: float = 22.0
    time: float = 0.0
    obstacles: list[Box] = field(default_factory=list)

    def copy(self) -> "LabState":
        return copy.deepcopy(self)

    # ----------------------------------------------------------- lookups
    def entity(self, eid: str | None) -> Container | Equipment | None:
        if eid is None:
            return None
        return self.containers.get(eid) or self.equipment.get(eid)

    def is_zone(self, name: str | None) -> bool:
        return name is not None and name in self.zones

    def in_zone(self, pos: np.ndarray, zone: str, margin: float = 0.0) -> bool:
        if zone not in self.zones:
            return False
        lo, hi = (np.asarray(b, dtype=float) for b in self.zones[zone])
        p = np.asarray(pos, dtype=float)
        return bool(np.all(p >= lo - margin) and np.all(p <= hi + margin))

    def zone_center(self, zone: str) -> np.ndarray:
        lo, hi = (np.asarray(b, dtype=float) for b in self.zones[zone])
        c = (lo + hi) / 2.0
        c[2] = lo[2]
        return c

    def zones_at(self, pos: np.ndarray) -> list[str]:
        return [z for z in self.zones if self.in_zone(pos, z)]

    def containers_in_zone(self, zone: str, exclude: str | None = None) -> list[Container]:
        return [c for c in self.containers.values()
                if c.id != exclude and self.in_zone(c.position, zone)]

    def equipment_of_kind(self, kind: str) -> list[Equipment]:
        return [e for e in self.equipment.values() if e.kind == kind]

    def iter_entities(self) -> Iterator[Container | Equipment]:
        yield from self.containers.values()
        yield from self.equipment.values()

    def mark_done(self, step: str) -> None:
        if step not in self.done_steps:
            self.done_steps.append(step)

    def summary(self) -> dict:
        return {
            "time": round(self.time, 3),
            "containers": {c.id: {"contents": {k: round(v, 2) for k, v in c.contents.items()},
                                  "T": round(c.temperature, 1), "lid_open": c.lid_open,
                                  "pos": np.round(c.position, 3).tolist()}
                           for c in self.containers.values()},
            "robot": {"ee": np.round(self.robot.ee_pos, 3).tolist(), "holding": self.robot.holding},
            "ppe": sorted(self.ppe),
        }
