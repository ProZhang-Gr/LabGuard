"""Evaluation context: binds predicate names (src, dst, action, robot, env and
the function vocabulary) to the current state s_t and proposed action a_t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from labguard.core.action import Action
from labguard.core.state import Container, Equipment, LabState
from labguard.ir.registry import Registry
from labguard.vocab import PLACEMENT_TAGS, TRANSFER_TAGS

WASTE_FULL_RATIO = 0.9


# ----------------------------------------------------------------------------
# Views: what `src.xxx` / `dst.xxx` resolve to
# ----------------------------------------------------------------------------
@dataclass
class ContainerView:
    id: str
    kind: str | None
    substances: frozenset[str]
    volume: float | None = None
    capacity: float | None = None
    temperature: float | None = None
    fill_ratio: float | None = None
    lid_open: bool | None = None
    open_time: float | None = None
    label: str | None = None

    @classmethod
    def of_container(cls, c: Container) -> "ContainerView":
        return cls(c.id, c.kind, frozenset(c.substances), c.volume, c.capacity, c.temperature,
                   c.fill_ratio, c.lid_open if c.has_lid else True, c.open_time, c.label)

    @classmethod
    def of_zone(cls, state: LabState, zone: str, exclude: str | None) -> "ContainerView":
        subs: set[str] = set()
        for c in state.containers_in_zone(zone, exclude=exclude):
            subs |= c.substances
        return cls(zone, "zone", frozenset(subs), label=zone)

    @classmethod
    def of_equipment(cls, e: Equipment) -> "ContainerView":
        return cls(e.id, e.kind, frozenset(), temperature=e.setpoint)


@dataclass
class ActionView:
    tag: str
    volume: float | None
    speed: float | None
    pour_rate: float | None
    target_temperature: float | None
    grip_force: float | None
    tilt_angle: float | None
    duration: float | None
    fill_after: float | None


@dataclass
class RobotView:
    speed: float | None
    holding: str | None
    gripper_force: float | None


@dataclass
class EnvView:
    ambient_temperature: float | None
    time: float | None


# ----------------------------------------------------------------------------
def operation_position(state: LabState, action: Action) -> np.ndarray:
    """Where the action takes place (used by in_zone / distance)."""
    if action.target_pos is not None:
        return np.asarray(action.target_pos, dtype=float)
    if action.dst is not None:
        if action.dst in state.containers:
            return state.containers[action.dst].position
        if action.dst in state.equipment:
            return state.equipment[action.dst].position
        if state.is_zone(action.dst):
            return state.zone_center(action.dst)
    if action.src is not None and action.src in state.containers:
        return state.containers[action.src].position
    return state.robot.ee_pos


def fill_after(state: LabState, action: Action) -> float | None:
    if action.tag not in TRANSFER_TAGS or action.volume is None:
        return None
    dst = state.containers.get(action.dst) if action.dst else None
    if dst is None or dst.capacity <= 0:
        return None
    return (dst.volume + action.volume) / dst.capacity


class EvalContext:
    """Namespace factory for Predicate.evaluate()."""

    def __init__(self, state: LabState, action: Action, registry: Registry,
                 substances: tuple[str, ...] = ()) -> None:
        self.state = state
        self.action = action
        self.registry = registry
        self.scope_chems = frozenset(s for s in substances if registry.is_chemical(s))
        self.pos = operation_position(state, action)

    # ----------------------------------------------------------------- views
    def src_view(self) -> ContainerView | None:
        c = self.state.containers.get(self.action.src) if self.action.src else None
        return ContainerView.of_container(c) if c else None

    def dst_view(self) -> ContainerView | None:
        a, s = self.action, self.state
        if a.dst is None:
            return None
        if a.dst in s.containers:
            return ContainerView.of_container(s.containers[a.dst])
        if a.dst in s.equipment:
            return ContainerView.of_equipment(s.equipment[a.dst])
        if s.is_zone(a.dst):
            return ContainerView.of_zone(s, a.dst, exclude=a.src)
        return None

    def action_view(self) -> ActionView:
        a = self.action
        return ActionView(a.tag, a.volume, a.speed, a.pour_rate, a.target_temperature, a.grip_force,
                          a.tilt_angle, a.duration, fill_after(self.state, a))

    def namespace(self) -> dict[str, Any]:
        r = self.state.robot
        return {
            "src": self.src_view(),
            "dst": self.dst_view(),
            "action": self.action_view(),
            "robot": RobotView(r.speed, r.holding, r.gripper_force),
            "env": EnvView(self.state.ambient_temperature, self.state.time),
            "__functions__": {
                "contains": self.f_contains,
                "incompatible": self.f_incompatible,
                "hazard": self.f_hazard,
                "has_ppe": self.f_has_ppe,
                "done": self.f_done,
                "in_zone": self.f_in_zone,
                "status": self.f_status,
                "distance": self.f_distance,
                "abs": abs,
                "min": min,
                "max": max,
            },
        }

    # ------------------------------------------------------------- functions
    def f_contains(self, view: ContainerView, sub: str) -> bool:
        sid = self.registry.resolve(sub) or sub
        return sid in view.substances

    def incompatible_pairs(self, a: ContainerView, b: ContainerView) -> list[tuple[str, str]]:
        """S-restricted incompatibility (Eq. 7): with >=2 chemicals in S both
        members of a pair must be in S, with one chemical at least one, with
        none the full table applies."""
        out = []
        for x in a.substances:
            for y in b.substances:
                if self.registry.compat(x, y):
                    continue
                k = len(self.scope_chems)
                in_s = (x in self.scope_chems) + (y in self.scope_chems)
                if (k >= 2 and in_s == 2) or (k == 1 and in_s >= 1) or k == 0:
                    out.append((x, y))
        return out

    def f_incompatible(self, a: ContainerView, b: ContainerView) -> bool:
        return bool(self.incompatible_pairs(a, b))

    def f_hazard(self, view: ContainerView, cls: str) -> bool:
        return any(self.registry.has_hazard(s, cls) for s in view.substances)

    def f_has_ppe(self, item: str) -> bool:
        return item in self.state.ppe

    def f_done(self, step: str) -> bool:
        return step in self.state.done_steps

    def f_in_zone(self, zone: str) -> bool | None:
        if zone not in self.state.zones:
            return None
        a = self.action
        if a.target_pos is None and self.state.is_zone(a.dst) and a.tag in PLACEMENT_TAGS | {"transport", "move"}:
            if a.dst == zone:
                return True
        return self.state.in_zone(self.pos, zone)

    def _involved_container(self) -> Container | None:
        for cid in (self.action.src, self.action.dst):
            if cid and cid in self.state.containers:
                return self.state.containers[cid]
        return None

    def f_status(self, entity: str, flag: str) -> bool | None:
        kind = self.registry.resolve(entity) or entity
        if kind == "glassware":
            c = self._involved_container()
            if c is None:
                return None
            return {"intact": c.intact, "clean": c.clean, "dry": c.dry}.get(flag)
        if kind == "waste_container":
            c = self.state.containers.get(self.action.dst) if self.action.dst else None
            if c is None or c.kind != "waste_container":
                return None
            return {"labeled": c.label is not None, "closed": not c.lid_open,
                    "not_full": c.fill_ratio < WASTE_FULL_RATIO}.get(flag)
        candidates = self.state.equipment_of_kind(kind)
        if not candidates:
            return None
        target = next((e for e in candidates if e.id == self.action.dst), None)
        if target is None:  # nearest unit of that kind
            target = min(candidates, key=lambda e: float(np.linalg.norm(e.position - self.pos)))
        return target.flags.get(flag)

    def f_distance(self, entity: str) -> float | None:
        kind = self.registry.resolve(entity) or entity
        positions = [e.position for e in self.state.equipment_of_kind(kind)]
        if self.registry.is_chemical(kind):
            positions += [c.position for c in self.state.containers.values()
                          if kind in c.substances and c.id != self.action.src]
        if kind in self.state.zones:
            positions.append(self.state.zone_center(kind))
        if not positions:
            return None
        return float(min(np.linalg.norm(np.asarray(p) - self.pos) for p in positions))
