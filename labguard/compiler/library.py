"""Typed monitor library (Sec. 3.4).

Every constraint type lowers to one check class.  A check

  * evaluates the IR predicate on (s_t, a_t) through an EvalContext,
  * declares which predicate features it can lower (an IR whose predicate
    references none of them is rejected at the `lowering` stage),
  * optionally implements a typed *repair* used by `modify` interventions
    (clamp a temperature, reduce a volume, scale a velocity, re-route to a
    compatible zone / waste container, ...).

Repairs are always re-checked by the caller; a repair that does not restore
the requirement escalates to `block`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from labguard.core.action import Action
from labguard.core.context import EvalContext
from labguard.core.state import LabState
from labguard.ir.predicate import Comparison, Predicate
from labguard.ir.registry import Registry
from labguard.ir.schema import LabGuardIR
from labguard.core.franka import PandaKinematics
from labguard.vocab import MOTION_TAGS, PLACEMENT_TAGS

_STRICT_EPS = 1e-3
_KIN = PandaKinematics()
_DT = 0.05
STORAGE_ZONES: tuple[str, ...] = (
    "acid_cabinet", "base_cabinet", "flammable_cabinet", "oxidizer_shelf", "general_shelf",
)
_HAZARD_HOME = {"acid": "acid_cabinet", "base": "base_cabinet", "flammable": "flammable_cabinet",
                "oxidizer": "oxidizer_shelf"}


class BaseCheck:
    constraint_type: ClassVar[str] = ""
    # predicate features (attribute paths "root.attr" or function names)
    features: ClassVar[frozenset[str]] = frozenset()
    # action attributes this check may clamp in a `modify` repair
    repairable: ClassVar[tuple[str, ...]] = ()

    def __init__(self, ir: LabGuardIR, predicate: Predicate, registry: Registry) -> None:
        self.ir = ir
        self.predicate = predicate
        self.registry = registry
        self.scope_chems = tuple(s for s in ir.substances if registry.is_chemical(s))

    # ------------------------------------------------------------ lowering
    @classmethod
    def lowering_error(cls, predicate: Predicate) -> str | None:
        used = predicate.attributes | predicate.call_names()
        if cls.features and not (used & cls.features):
            return (f"{cls.__name__} cannot lower predicate {predicate.canonical!r}: it references none "
                    f"of {sorted(cls.features)}")
        return None

    @property
    def name(self) -> str:
        return type(self).__name__

    def theta(self) -> dict[str, Any]:
        """Monitor parameters theta (used for parameter-recovery diagnostics)."""
        limits = tuple(sorted((c.path, c.op, repr(c.value)) for c in self.predicate.comparisons))
        args = tuple(sorted((fn, tuple(repr(a) for a in a_s)) for fn, a_s in self.predicate.calls
                            if fn not in ("abs", "min", "max")))
        return {"limits": limits, "calls": args}

    # ----------------------------------------------------------- semantics
    def context(self, state: LabState, action: Action) -> EvalContext:
        return EvalContext(state, action, self.registry, self.ir.substances)

    def binds(self, state: LabState, action: Action) -> bool:
        """Substance binding: chemical members of S must be involved in the action.
        Entity-only S (glassware, hot plate, ...) always binds."""
        if not self.scope_chems:
            return True
        ctx = self.context(state, action)
        involved: set[str] = set()
        for view in (ctx.src_view(), ctx.dst_view()):
            if view is not None:
                involved |= view.substances
        if state.robot.holding and state.robot.holding in state.containers and action.src is None:
            involved |= state.containers[state.robot.holding].substances
        return any(c in involved for c in self.scope_chems)

    def evaluate(self, state: LabState, action: Action) -> bool | None:
        """True = requirement holds, False = violated, None = not applicable."""
        return self.predicate.evaluate(self.context(state, action).namespace())

    def explain(self, state: LabState, action: Action) -> str:
        return f"{self.ir.constraint_type} violated: requires `{self.predicate.canonical}`"

    # --------------------------------------------------------------- repair
    def repair(self, state: LabState, action: Action) -> Action | None:
        return self._clamp_repair(state, action)

    def _clamp_repair(self, state: LabState, action: Action) -> Action | None:
        limits = [c for c in self.predicate.comparisons
                  if c.path.startswith("action.") and c.path[7:] in self.repairable
                  and isinstance(c.value, (int, float)) and not isinstance(c.value, bool)]
        if not limits:
            return None
        new = action.copy()
        changed = False
        for c in limits:
            changed |= self._apply_limit(state, new, c)
        return new if changed else None

    def _apply_limit(self, state: LabState, action: Action, c: Comparison) -> bool:
        attr = c.path[7:]
        cur = getattr(action, attr, None)
        if cur is None:
            return False
        target = _clamp_value(cur, c.op, float(c.value))
        if target is None or target == cur:
            return False
        self._set(state, action, attr, cur, target)
        return True

    def _set(self, state: LabState, action: Action, attr: str, old: float, new: float) -> None:
        setattr(action, attr, new)


def _clamp_value(cur: float, op: str, limit: float) -> float | None:
    if op == "<=":
        return min(cur, limit)
    if op == "<":
        return cur if cur < limit else limit - _STRICT_EPS
    if op == ">=":
        return max(cur, limit)
    if op == ">":
        return cur if cur > limit else limit + _STRICT_EPS
    return None


# ============================================================================
# Material compatibility family
# ============================================================================
class MaterialCompatibilityCheck(BaseCheck):
    constraint_type = "chemical_incompatibility"
    features = frozenset({"incompatible", "contains", "hazard"})

    def explain(self, state: LabState, action: Action) -> str:
        ctx = self.context(state, action)
        src, dst = ctx.src_view(), ctx.dst_view()
        pairs = ctx.incompatible_pairs(src, dst) if src and dst else []
        if pairs:
            txt = ", ".join(f"{a}+{b}" for a, b in sorted(set(pairs)))
            return f"incompatible materials would be combined: {txt}"
        return super().explain(state, action)

    def repair(self, state: LabState, action: Action) -> Action | None:
        return None  # no safe way to partially mix incompatible chemicals


class StorageSegregationCheck(BaseCheck):
    constraint_type = "storage_constraint"
    features = frozenset({"incompatible", "hazard", "in_zone", "contains"})

    def repair(self, state: LabState, action: Action) -> Action | None:
        if action.tag not in PLACEMENT_TAGS | {"transport"}:
            return None
        src = state.containers.get(action.src) if action.src else None
        preferred = []
        if src is not None:
            for hz, zone in _HAZARD_HOME.items():
                if any(self.registry.has_hazard(s, hz) for s in src.substances):
                    preferred.append(zone)
        candidates = [z for z in (*preferred, *STORAGE_ZONES) if z in state.zones and z != action.dst]
        for zone in dict.fromkeys(candidates):
            new = action.copy()
            new.dst = zone
            if new.target_pos is not None:
                new.target_pos = state.zone_center(zone) + np.array([0.0, 0.0, float(action.target_pos[2])])
            if self.evaluate(state, new) is not False:
                return new
        return None


# ============================================================================
# Physical / quantitative family
# ============================================================================
class TemperatureLimitCheck(BaseCheck):
    constraint_type = "temperature_constraint"
    features = frozenset({"action.target_temperature", "src.temperature", "dst.temperature",
                          "env.ambient_temperature"})
    repairable = ("target_temperature",)


class ContainerCapacityCheck(BaseCheck):
    constraint_type = "container_capacity"
    features = frozenset({"action.fill_after", "action.volume", "dst.fill_ratio", "dst.volume",
                          "src.fill_ratio", "src.volume"})
    repairable = ("fill_after", "volume")

    def _apply_limit(self, state: LabState, action: Action, c: Comparison) -> bool:
        if c.path != "action.fill_after":
            return super()._apply_limit(state, action, c)
        dst = state.containers.get(action.dst) if action.dst else None
        if dst is None or action.volume is None or c.op not in ("<=", "<"):
            return False
        margin = _STRICT_EPS if c.op == "<" else 0.0
        allowed = (float(c.value) - margin) * dst.capacity - dst.volume
        if allowed <= 0.0:
            action.volume = 0.0
            return True
        if action.volume <= allowed:
            return False
        action.volume = allowed
        return True

    def repair(self, state: LabState, action: Action) -> Action | None:
        new = super().repair(state, action)
        if new is None or not new.volume or new.volume <= 0:
            return None  # nothing can be transferred safely -> escalate to block
        return new


class VelocityLimitCheck(BaseCheck):
    constraint_type = "velocity_limit"
    features = frozenset({"action.speed", "robot.speed"})
    repairable = ("speed",)

    def _set(self, state: LabState, action: Action, attr: str, old: float, new: float) -> None:
        setattr(action, attr, new)
        if attr == "speed" and old > 0:
            scale_motion(state, action, new / old)


def scale_motion(state: LabState, action: Action, factor: float) -> None:
    """Scale the low-level part of an action by `factor` (velocity repair)."""
    if action.qdot is not None:
        action.qdot = action.qdot * factor
    if action.q_target is not None:
        q = state.robot.q
        action.q_target = q + (action.q_target - q) * factor
    if action.target_pos is not None and action.tag in MOTION_TAGS:
        ee = state.robot.ee_pos
        action.target_pos = ee + (action.target_pos - ee) * factor
    if action.duration is not None and factor > 0:
        action.duration = action.duration / factor


class ForceLimitCheck(BaseCheck):
    constraint_type = "force_limit"
    features = frozenset({"action.grip_force", "robot.gripper_force"})
    repairable = ("grip_force",)


class WorkspaceZoneCheck(BaseCheck):
    constraint_type = "workspace_zone"
    features = frozenset({"in_zone", "distance"})
    margin = 0.01

    def repair(self, state: LabState, action: Action) -> Action | None:
        if action.target_pos is None:
            return None
        zones = [args[0] for fn, args in self.predicate.calls if fn == "in_zone"]
        negated = "not in_zone(" in self.predicate.canonical
        if not zones or negated:
            return None
        new = action.copy()
        for zone in zones:
            lo, hi = (np.asarray(b, dtype=float) for b in state.zones[zone])
            new.target_pos = np.clip(new.target_pos, lo + self.margin, hi - self.margin)
        if new.q_target is not None:  # keep the joint command consistent with the projected target
            new.q_target, _ = _KIN.ik_position(new.target_pos, new.q_target)
            new.qdot = None if action.qdot is None else (new.q_target - state.robot.q) / _DT
        return new


class SpillPreventionCheck(BaseCheck):
    constraint_type = "spill_prevention"
    features = frozenset({"action.pour_rate", "action.tilt_angle", "action.speed", "action.volume",
                          "src.lid_open", "src.fill_ratio", "dst.fill_ratio", "action.fill_after"})
    repairable = ("pour_rate", "tilt_angle", "speed")

    def _set(self, state: LabState, action: Action, attr: str, old: float, new: float) -> None:
        setattr(action, attr, new)
        if attr == "speed" and old > 0:
            scale_motion(state, action, new / old)


# ============================================================================
# Process / equipment family
# ============================================================================
class EquipmentStatusCheck(BaseCheck):
    constraint_type = "equipment_check"
    features = frozenset({"status"})

    def repair(self, state: LabState, action: Action) -> Action | None:
        return None


class SequencePreconditionCheck(BaseCheck):
    constraint_type = "sequence_constraint"
    features = frozenset({"done", "src.lid_open", "dst.lid_open", "contains", "status"})

    def repair(self, state: LabState, action: Action) -> Action | None:
        return None


# ============================================================================
# Environment / PPE / waste family
# ============================================================================
class ExposureLimitCheck(BaseCheck):
    constraint_type = "exposure_limit"
    features = frozenset({"src.open_time", "dst.open_time", "action.duration", "in_zone", "has_ppe"})
    repairable = ("duration",)


class VentilationCheck(BaseCheck):
    constraint_type = "ventilation_requirement"
    features = frozenset({"in_zone", "status"})

    def repair(self, state: LabState, action: Action) -> Action | None:
        return None


class FireSafetyCheck(BaseCheck):
    constraint_type = "fire_safety"
    features = frozenset({"distance", "status", "hazard", "in_zone", "contains",
                          "action.target_temperature"})
    repairable = ("target_temperature",)


class PPECheck(BaseCheck):
    constraint_type = "ppe_requirement"
    features = frozenset({"has_ppe"})

    def repair(self, state: LabState, action: Action) -> Action | None:
        return None


class WasteHandlingCheck(BaseCheck):
    constraint_type = "waste_handling"
    features = frozenset({"dst.label", "dst.kind", "incompatible", "contains", "hazard", "status"})

    def repair(self, state: LabState, action: Action) -> Action | None:
        if action.tag not in ("dispose", "pour", "transfer"):
            return None
        for c in state.containers.values():
            if c.kind != "waste_container" or c.id in (action.src, action.dst):
                continue
            new = action.copy()
            new.dst = c.id
            new.target_pos = None
            if self.evaluate(state, new) is True:
                return new
        return None


MONITOR_LIBRARY: dict[str, type[BaseCheck]] = {
    cls.constraint_type: cls
    for cls in (
        MaterialCompatibilityCheck, StorageSegregationCheck, TemperatureLimitCheck,
        ContainerCapacityCheck, VelocityLimitCheck, ForceLimitCheck, WorkspaceZoneCheck,
        SpillPreventionCheck, EquipmentStatusCheck, SequencePreconditionCheck, ExposureLimitCheck,
        VentilationCheck, FireSafetyCheck, PPECheck, WasteHandlingCheck,
    )
}

__all__ = ["BaseCheck", "MONITOR_LIBRARY", "scale_motion", *(
    c.__name__ for c in MONITOR_LIBRARY.values())]
