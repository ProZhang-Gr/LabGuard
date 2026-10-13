"""Outcome oracle of the toy laboratory (stand-in for LabUtopia physics).

`apply_action` advances the state by one control step and returns the
hazards that the executed action caused.  It is shared by the toy simulator,
the PlanChecker transition model and the stress-benchmark replay, so that
"hazard prevented" is judged by the same physics everywhere.

Thresholds are intentionally *less* conservative than the rules (e.g. the
rule caps pouring at 25 mL/s, splashing starts at 40 mL/s), mirroring how
written safety rules keep a margin to the physical failure point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from labguard.core.action import Action
from labguard.core.franka import Q_MAX, Q_MIN, QDOT_MAX, PandaKinematics
from labguard.core.state import Container, LabState
from labguard.ir.registry import Registry, default_registry

KNOCK_OVER_SPEED = 0.30  # m/s, carrying a filled container
SPLASH_RATE = 40.0  # mL/s
BREAK_FORCE = 30.0  # N on glassware
FIRE_DISTANCE = 0.20  # m between an open flammable and a hot plate that is on
CHCl3_EXPOSURE_TIME = 120.0  # s open outside the fume hood
VENTILATED_SUBSTANCES = frozenset({"NH3"})
GRASP_OFFSET = 0.05  # container origin below the fingertips
POUR_ALIGN_TOL = 0.10  # m, horizontal offset between source and receiver when pouring
GLASS_KINDS = frozenset({"beaker", "flask", "test_tube", "cylinder", "bottle"})

FATAL_HAZARDS = frozenset({
    "hazardous_reaction", "overflow", "breakage", "collision_env", "collision_table", "knock_over",
    "fire_risk", "violent_boiling",
})
NONFATAL_HAZARDS = frozenset({
    "splash", "overheat", "exposure", "ppe_exposure", "waste_misrouting", "joint_limit", "spill",
})
HAZARD_CATEGORY: dict[str, str] = {
    "hazardous_reaction": "incompatible_mixing",
    "overflow": "overflow",
    "overheat": "overheating",
    "fire_risk": "overheating",
    "violent_boiling": "sequence_violation",
    "spill": "sequence_violation",
    "exposure": "ppe_ventilation",
    "ppe_exposure": "ppe_ventilation",
    "waste_misrouting": "waste_misrouting",
    "knock_over": "motion_hazard",
    "collision_env": "motion_hazard",
    "collision_table": "motion_hazard",
    "joint_limit": "motion_hazard",
    "breakage": "motion_hazard",
    "splash": "motion_hazard",
}


@dataclass(frozen=True)
class Hazard:
    kind: str
    detail: str = ""
    t: float = 0.0

    @property
    def fatal(self) -> bool:
        return self.kind in FATAL_HAZARDS

    @property
    def category(self) -> str:
        return HAZARD_CATEGORY.get(self.kind, "other")


# ----------------------------------------------------------------------------
# Kinematic part
# ----------------------------------------------------------------------------
_KIN = PandaKinematics()


def apply_motion(state: LabState, action: Action, dt: float, kin: PandaKinematics = _KIN) -> list[Hazard]:
    r = state.robot
    if action.q_target is not None:
        q_new = np.asarray(action.q_target, dtype=float)
    elif action.qdot is not None:
        q_new = r.q + np.asarray(action.qdot, dtype=float) * dt
    elif action.target_pos is not None:
        q_new, _ = kin.ik_position(action.target_pos, r.q)
    else:
        return []
    hz: list[Hazard] = []
    qdot = (q_new - r.q) / dt
    if np.any(q_new < Q_MIN) or np.any(q_new > Q_MAX) or np.any(np.abs(qdot) > QDOT_MAX * 1.0001):
        hz.append(Hazard("joint_limit", "joint position/velocity limit exceeded", state.time))
        q_new = np.clip(q_new, Q_MIN, Q_MAX)
        qdot = np.clip((q_new - r.q) / dt, -QDOT_MAX, QDOT_MAX)
        q_new = r.q + qdot * dt
    ee_old = r.ee_pos.copy()
    ee = kin.position(q_new)
    r.q, r.qdot, r.ee_pos = q_new, qdot, ee
    r.speed = float(np.linalg.norm(ee - ee_old) / dt)
    held = state.containers.get(r.holding) if r.holding else None
    if ee[2] < 0.0:
        hz.append(Hazard("collision_table", f"end effector at z={ee[2]:.3f}", state.time))
    for lo, hi in state.obstacles:
        if np.all(ee >= np.asarray(lo)) and np.all(ee <= np.asarray(hi)):
            hz.append(Hazard("collision_env", "end effector entered an obstacle", state.time))
            break
    if held is not None:
        held.position = ee - np.array([0.0, 0.0, GRASP_OFFSET])
        if r.speed > KNOCK_OVER_SPEED and held.volume > 0 and not (held.has_lid and not held.lid_open):
            spilled = held.remove(held.volume * 0.5)
            hz.append(Hazard("knock_over", f"{held.id} spilled {sum(spilled.values()):.0f} mL at "
                                           f"{r.speed:.2f} m/s", state.time))
    return hz


# ----------------------------------------------------------------------------
# Semantic part
# ----------------------------------------------------------------------------
def _exposure_check(state: LabState, c: Container, pos: np.ndarray) -> list[Hazard]:
    hz = []
    if c.substances & VENTILATED_SUBSTANCES and not state.in_zone(pos, "fume_hood"):
        hz.append(Hazard("exposure", f"{c.id}: volatile toxic vapour outside the fume hood", state.time))
    return hz


def _ppe_check(state: LabState, registry: Registry, containers: list[Container | None]) -> list[Hazard]:
    if "gloves" in state.ppe:
        return []
    for c in containers:
        if c is not None and any(registry.has_hazard(s, "corrosive") for s in c.substances):
            return [Hazard("ppe_exposure", f"handling corrosive {c.id} without gloves", state.time)]
    return []


def _fire_check(state: LabState, registry: Registry, c: Container | None) -> list[Hazard]:
    if c is None or not any(registry.has_hazard(s, "flammable") for s in c.substances):
        return []
    for e in state.equipment_of_kind("hot_plate"):
        if e.flags.get("on") and np.linalg.norm(c.position[:2] - e.position[:2]) < FIRE_DISTANCE:
            return [Hazard("fire_risk", f"flammable {c.id} next to a hot plate", state.time)]
    return []


def _transfer(state: LabState, registry: Registry, action: Action, src: Container,
              dst: Container) -> list[Hazard]:
    hz: list[Hazard] = []
    if src.has_lid and not src.lid_open:
        return [Hazard("spill", f"tilting closed {src.id}: cap leaks", state.time)]
    vol = min(float(action.volume if action.volume is not None else src.volume), src.volume)
    if vol <= 0:
        return hz
    rate = float(action.pour_rate or 15.0)
    if np.linalg.norm(src.position[:2] - dst.position[:2]) > POUR_ALIGN_TOL:
        src.remove(vol)
        return [Hazard("spill", f"poured {vol:.0f} mL beside {dst.id}", state.time)]
    mixture = src.remove(vol)
    for a in mixture:
        for b in dst.substances:
            if not registry.compat(a, b):
                hz.append(Hazard("hazardous_reaction", f"{a} + {b} in {dst.id}", state.time))
    if "H2O" in mixture and dst.contents.get("H2SO4", 0.0) > 0.5 * dst.volume > 0:
        hz.append(Hazard("violent_boiling", f"water added to concentrated H2SO4 in {dst.id}", state.time))
    dst.add(mixture)
    if dst.volume > dst.capacity + 1e-6:
        excess = dst.volume - dst.capacity
        dst.remove(excess)
        hz.append(Hazard("overflow", f"{dst.id} overflowed by {excess:.0f} mL", state.time))
    if rate > SPLASH_RATE:
        hz.append(Hazard("splash", f"pour rate {rate:.0f} mL/s", state.time))
    if dst.kind == "waste_container":
        for s in mixture:
            stream = registry.waste_stream(s)
            if s != "H2O" and stream != dst.label:
                hz.append(Hazard("waste_misrouting", f"{s} into {dst.label or 'unlabelled'} container",
                                 state.time))
                break
    hz += _exposure_check(state, src, dst.position)
    hz += _ppe_check(state, registry, [src, dst])
    return hz


def apply_semantic(state: LabState, action: Action, registry: Registry | None = None,
                   dt: float = 0.05) -> list[Hazard]:
    """Semantic (skill-level) effect of an action; also used by the PlanChecker."""
    registry = registry or default_registry()
    tag = action.tag
    src = state.containers.get(action.src) if action.src else None
    dst_c = state.containers.get(action.dst) if action.dst else None
    dst_e = state.equipment.get(action.dst) if action.dst else None
    r = state.robot
    hz: list[Hazard] = []
    duration = float(action.duration) if action.duration else dt
    plan_level = action.q_target is None and action.qdot is None

    if tag in ("pour", "transfer", "dispense", "mix", "dispose"):
        if src is not None and dst_c is not None:
            hz += _transfer(state, registry, action, src, dst_c)
    elif tag == "heat":
        if src is not None:
            target = float(action.target_temperature if action.target_temperature is not None else 60.0)
            if not src.intact:
                hz.append(Hazard("breakage", f"cracked {src.id} shattered on heating", state.time))
            elif src.has_lid and not src.lid_open and target > 50.0:
                hz.append(Hazard("breakage", f"sealed {src.id} over-pressurised", state.time))
            src.temperature = target
            for s in src.substances:
                limit = registry.max_safe_temp(s)
                if limit is not None and target > limit:
                    kind = "fire_risk" if registry.has_hazard(s, "flammable") else "overheat"
                    hz.append(Hazard(kind, f"{s} heated to {target:.0f} C (> {limit:.0f} C)", state.time))
            if dst_e is not None:
                dst_e.setpoint = target
                dst_e.flags["on"] = True
    elif tag == "shake":
        if src is not None and src.lid_open and src.volume > 0:
            src.remove(src.volume * 0.2)
            hz.append(Hazard("spill", f"shaking open {src.id}", state.time))
            hz += _exposure_check(state, src, src.position)
    elif tag == "open":
        if src is not None and src.has_lid and not src.lid_open:
            src.lid_open = True
            src.open_time = 0.0
            hz += _exposure_check(state, src, src.position)
    elif tag == "close":
        if src is not None:
            src.lid_open = False
    elif tag in ("press", "centrifuge"):
        if dst_e is not None:
            if dst_e.kind == "centrifuge" and not dst_e.flags.get("balanced", True):
                hz.append(Hazard("breakage", "unbalanced centrifuge rotor failure", state.time))
            dst_e.flags["on"] = not dst_e.flags.get("on", False) if tag == "press" else True
    elif tag == "pick":
        if src is not None and action.grip_force is not None:
            r.gripper_force = float(action.grip_force)
            if src.kind in GLASS_KINDS and action.grip_force > BREAK_FORCE:
                src.intact = False
                hz.append(Hazard("breakage", f"{src.id} crushed with {action.grip_force:.0f} N", state.time))
            r.holding = src.id
    elif tag in ("place", "store"):
        held = src or (state.containers.get(r.holding) if r.holding else None)
        if held is not None:
            if action.target_pos is not None:
                held.position = np.asarray(action.target_pos, dtype=float) - np.array([0, 0, GRASP_OFFSET])
                held.position[2] = max(0.0, held.position[2])
            elif state.is_zone(action.dst):
                held.position = state.zone_center(action.dst)  # type: ignore[arg-type]
            if action.gripper is None or action.gripper > 0 or plan_level:
                if r.holding == held.id:
                    r.holding = None
                    r.gripper_force = 0.0
            hz += _fire_check(state, registry, held)
    elif tag in ("transport", "move"):
        if plan_level and src is not None:  # skill-level transport: teleport to destination
            if action.target_pos is not None:
                src.position = np.asarray(action.target_pos, dtype=float)
            elif state.is_zone(action.dst):
                src.position = state.zone_center(action.dst)  # type: ignore[arg-type]
            elif dst_c is not None:
                src.position = dst_c.position + np.array([0.0, 0.0, 0.15])
            if action.speed is not None and action.speed > KNOCK_OVER_SPEED and src.volume > 0:
                hz.append(Hazard("knock_over", f"{src.id} transported at {action.speed:.2f} m/s", state.time))
        held = state.containers.get(r.holding) if r.holding else src
        hz += _fire_check(state, registry, held)
    elif tag == "clean":
        if src is not None:
            src.clean = src.volume <= 1e-6
    elif tag == "inspect":
        pass

    # exposure from containers left open
    for c in state.containers.values():
        if c.has_lid and c.lid_open:
            c.open_time += duration
            if "CHCl3" in c.substances and c.open_time > CHCl3_EXPOSURE_TIME \
                    and not state.in_zone(c.position, "fume_hood") and c.open_time - duration <= CHCl3_EXPOSURE_TIME:
                hz.append(Hazard("exposure", f"{c.id} chloroform open for {c.open_time:.0f} s", state.time))
    if tag not in ("move",):
        state.mark_done(tag)
    state.time += duration
    return hz


def apply_action(state: LabState, action: Action, registry: Registry | None = None, dt: float = 0.05,
                 kin: PandaKinematics = _KIN) -> list[Hazard]:
    """Full control step: robot motion (if commanded) followed by semantic effects."""
    hz = apply_motion(state, action, dt, kin)
    if action.duration is None:
        action = action.copy()
        action.duration = dt
    hz += apply_semantic(state, action, registry, dt)
    return hz
