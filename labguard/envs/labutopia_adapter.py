"""Adapter that inserts LabGuard between a LabUtopia controller and the robot.

LabUtopia's main loop (main.py) does

    state = task.step()
    action, done, is_success = task_controller.step(state)
    robot.get_articulation_controller().apply_action(action)

`SafeLabUtopiaController` wraps `task_controller` so that every
ArticulationAction is mapped to a LabGuard `Action`, filtered by the
SafetyShield and mapped back (blocked actions become a "hold" command).

LabUtopia does not simulate chemistry, so the adapter keeps a *shadow*
LabState: object poses and joint positions come from the task state dict
(keys used by LabUtopia controllers: object_position, object_name,
object_size, target_position, joint_positions, gripper_position, ...),
while container contents come from configurable object profiles and are
updated with the semantic transition model when a pour phase completes.

Isaac Sim is imported lazily, so this module can be unit-tested without it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from labguard.core.action import Action
from labguard.core.franka import Q_HOME, PandaKinematics
from labguard.core.state import Container, Equipment, LabState, RobotState
from labguard.envs.lab_physics import apply_semantic
from labguard.ir.registry import Registry, default_registry
from labguard.runtime.executor import StepDecision
from labguard.runtime.shield import SafetyShield

# phase / controller names (lower-case substrings) -> LabGuard task tag
PHASE_TAGS: tuple[tuple[str, str], ...] = (
    ("pick", "pick"), ("place", "place"), ("pour", "pour"), ("shake", "shake"), ("stir", "stir"),
    ("open", "open"), ("close", "close"), ("press", "press"), ("heat", "heat"),
    ("transport", "transport"), ("move", "transport"), ("clean", "clean"), ("device", "press"),
)

# object-name keyword -> shadow container profile
DEFAULT_OBJECT_PROFILES: dict[str, dict[str, Any]] = {
    "conical_bottle": {"kind": "flask", "contents": {"H2O": 100.0}, "capacity": 250.0},
    "beaker": {"kind": "beaker", "contents": {"H2O": 100.0}, "capacity": 250.0},
    "graduated_cylinder": {"kind": "cylinder", "contents": {"H2O": 50.0}, "capacity": 100.0},
    "bottle": {"kind": "bottle", "contents": {"H2O": 200.0}, "capacity": 500.0, "has_lid": True},
    "flask": {"kind": "flask", "contents": {"H2O": 100.0}, "capacity": 250.0},
}
DEFAULT_TARGET_PROFILE: dict[str, Any] = {"kind": "beaker", "contents": {}, "capacity": 250.0}


@dataclass
class LabUtopiaAdapterConfig:
    robot_base: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world position of the Franka base (cfg.robot.position)
    table_z: float | None = None  # world z of the bench top; default: robot base height
    dt: float = 1.0 / 60.0  # physics / control period of LabUtopia
    lookahead_steps: int = 5  # PD tracking horizon used to turn joint targets into velocities
    controller_type: str = ""
    object_profiles: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(DEFAULT_OBJECT_PROFILES))
    target_profile: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_TARGET_PROFILE))
    pour_volume: float | None = None  # planned transfer volume (default: everything in the source)
    pour_rate_per_rad_s: float = 20.0  # mL/s per rad/s of wrist rotation while pouring
    zones: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None
    ppe: tuple[str, ...] = ("gloves", "goggles", "lab_coat")
    equipment: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> {kind, flags, position}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "LabUtopiaAdapterConfig":
        d = dict(d or {})
        for key in ("robot_base", "ppe"):
            if key in d:
                d[key] = tuple(d[key])
        if d.get("zones"):
            d["zones"] = {k: (tuple(v[0]), tuple(v[1])) for k, v in d["zones"].items()}
        return cls(**d)


def phase_to_tag(phase: str | None, controller_type: str = "") -> str:
    for text in (phase or "", controller_type or ""):
        low = text.lower()
        for key, tag in PHASE_TAGS:
            if key in low:
                return tag
    return "move"


def _phase_name(controller: Any) -> str | None:
    phase = getattr(controller, "current_phase", None)
    if phase is None:
        return None
    return str(getattr(phase, "value", getattr(phase, "name", phase)))


def _clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(name).lower()).strip("_") or "object"


class LabUtopiaStateExtractor:
    """Builds the shadow LabState from LabUtopia's task state dict."""

    def __init__(self, config: LabUtopiaAdapterConfig, registry: Registry | None = None,
                 kin: PandaKinematics | None = None) -> None:
        self.cfg = config
        self.registry = registry or default_registry()
        self.kin = kin or PandaKinematics()
        self.base = np.asarray(config.robot_base, dtype=float)
        self.table_z = float(config.table_z if config.table_z is not None else config.robot_base[2])
        self.shadow: LabState | None = None
        self.source_id: str | None = None
        self.target_id: str | None = None

    def reset(self) -> None:
        self.shadow = None
        self.source_id = self.target_id = None

    def to_base(self, p_world: Any) -> np.ndarray:
        p = np.asarray(p_world, dtype=float)[:3] - self.base
        p[2] -= self.table_z - self.base[2]  # bench top at z = 0
        return p

    def _profile(self, name: str) -> dict[str, Any]:
        low = name.lower()
        for key, prof in self.cfg.object_profiles.items():
            if key in low:
                return prof
        return {"kind": "beaker", "contents": {}, "capacity": 250.0}

    def _container(self, cid: str, prof: dict[str, Any], pos: np.ndarray) -> Container:
        return Container(cid, prof.get("kind", "beaker"), dict(prof.get("contents", {})),
                         float(prof.get("capacity", 250.0)), has_lid=bool(prof.get("has_lid", False)),
                         lid_open=bool(prof.get("lid_open", True)), label=prof.get("label"), position=pos)

    def _init_shadow(self, state: dict[str, Any]) -> LabState:
        containers: dict[str, Container] = {}
        if "object_name" in state or "object_position" in state:
            self.source_id = _clean_name(state.get("object_name", "source"))
            containers[self.source_id] = self._container(
                self.source_id, self._profile(self.source_id), self.to_base(state.get("object_position", [0, 0, 0])))
        if "target_position" in state:
            self.target_id = _clean_name(state.get("target_name", "target"))
            if self.target_id == self.source_id:
                self.target_id += "_target"
            containers[self.target_id] = self._container(
                self.target_id, self.cfg.target_profile, self.to_base(state["target_position"]))
        equipment = {eid: Equipment(eid, e.get("kind", eid), dict(e.get("flags", {})),
                                    position=e.get("position", (0.0, 0.0, 0.0)))
                     for eid, e in self.cfg.equipment.items()}
        kwargs: dict[str, Any] = {}
        if self.cfg.zones:
            kwargs["zones"] = dict(self.cfg.zones)
        return LabState(containers=containers, equipment=equipment,
                        robot=RobotState(q=Q_HOME.copy(), ee_pos=self.kin.position(Q_HOME)),
                        ppe=set(self.cfg.ppe), **kwargs)

    def __call__(self, state: dict[str, Any], held: bool = False) -> LabState:
        if self.shadow is None:
            self.shadow = self._init_shadow(state)
        s = self.shadow
        q = np.asarray(state.get("joint_positions", Q_HOME), dtype=float)[:7]
        r = s.robot
        r.q = q
        r.ee_pos = self.kin.position(q)
        if "joint_velocities" in state:
            r.qdot = np.asarray(state["joint_velocities"], dtype=float)[:7]
            r.speed = self.kin.ee_speed(q, r.qdot)
        if self.source_id and "object_position" in state:
            s.containers[self.source_id].position = self.to_base(state["object_position"])
        if self.target_id and "target_position" in state:
            s.containers[self.target_id].position = self.to_base(state["target_position"])
        r.holding = self.source_id if held else None
        return s

    def complete_phase(self, tag: str, volume: float | None = None) -> None:
        """Apply the semantic effect of a finished phase to the shadow chemistry."""
        if self.shadow is None or tag not in ("pour", "heat", "open", "close", "shake", "stir", "clean"):
            return
        a = Action(tag, self.source_id, self.target_id if tag == "pour" else None, volume=volume)
        apply_semantic(self.shadow, a, self.registry, self.cfg.dt)


class LabUtopiaActionBridge:
    """ArticulationAction <-> LabGuard Action."""

    def __init__(self, config: LabUtopiaAdapterConfig, kin: PandaKinematics | None = None) -> None:
        self.cfg = config
        self.kin = kin or PandaKinematics()

    @staticmethod
    def _joint_array(values: Any, fallback: np.ndarray) -> np.ndarray:
        if values is None:
            return fallback.copy()
        out = fallback.copy()
        vals = list(np.asarray(values, dtype=object).ravel())[:7]
        for i, v in enumerate(vals):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out[i] = float(v)
        return out

    def to_labguard(self, art_action: Any, shadow: LabState, tag: str, src: str | None,
                    dst: str | None) -> Action:
        q = shadow.robot.q
        q_target = self._joint_array(getattr(art_action, "joint_positions", None), q)
        lookahead = self.cfg.dt * max(self.cfg.lookahead_steps, 1)
        jv = getattr(art_action, "joint_velocities", None)
        qdot = self._joint_array(jv, np.zeros(7)) if jv is not None else (q_target - q) / lookahead
        a = Action(tag, src, dst, q_target=q_target, qdot=qdot, target_pos=self.kin.position(q_target),
                   meta={"source": "labutopia", "lookahead": lookahead})
        a.speed = self.kin.ee_speed(q, qdot)
        if tag == "pour":
            src_c = shadow.containers.get(src or "")
            a.volume = self.cfg.pour_volume if self.cfg.pour_volume is not None else (src_c.volume if src_c else None)
            a.pour_rate = self.cfg.pour_rate_per_rad_s * abs(float(qdot[6]))
        return a

    def to_articulation(self, action: Action | None, original: Any, shadow: LabState) -> Any:
        """Write the filtered command back; `None` (blocked) becomes a hold command."""
        q_cmd = shadow.robot.q if action is None or action.q_target is None else action.q_target
        orig = getattr(original, "joint_positions", None)
        n = len(orig) if orig is not None else 9
        joints = [None] * n if orig is None else list(np.asarray(orig, dtype=object).ravel())
        for i in range(min(7, n)):
            joints[i] = float(q_cmd[i])
        if action is None:  # hold: keep the gripper fingers where they are
            for i in range(7, n):
                joints[i] = None
        return make_articulation_action(joint_positions=joints)


def make_articulation_action(joint_positions: list[Any]) -> Any:
    try:
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore[import-not-found]
    except ImportError:  # unit tests / no Isaac Sim
        from types import SimpleNamespace

        return SimpleNamespace(joint_positions=np.asarray(joint_positions, dtype=object),
                               joint_velocities=None, joint_efforts=None)
    arr = np.array([np.nan if v is None else v for v in joint_positions], dtype=float)
    return ArticulationAction(joint_positions=arr)


class SafeLabUtopiaController:
    """Drop-in replacement for a LabUtopia task controller with the LabGuard runtime in the loop."""

    def __init__(self, controller: Any, shield: SafetyShield | None = None,
                 config: LabUtopiaAdapterConfig | None = None, registry: Registry | None = None) -> None:
        self.controller = controller
        self.cfg = config or LabUtopiaAdapterConfig()
        if not self.cfg.controller_type:
            self.cfg.controller_type = type(controller).__name__
        self.shield = shield or SafetyShield.from_rules(registry=registry)
        if abs(self.shield.config.filter.dt - self.cfg.dt * self.cfg.lookahead_steps) > 1e-9:
            self.shield.config.filter.dt = self.cfg.dt * self.cfg.lookahead_steps
        self.extractor = LabUtopiaStateExtractor(self.cfg, registry)
        self.bridge = LabUtopiaActionBridge(self.cfg)
        self.decisions: list[StepDecision] = []
        self.steps = 0
        self.interventions = 0
        self._phase: str | None = None
        self._held = False

    # LabUtopia's main loop also calls reset/need_reset/episode_num/close/... -> delegate
    def __getattr__(self, name: str) -> Any:
        return getattr(self.controller, name)

    def _update_phase(self, phase: str | None) -> str:
        tag = phase_to_tag(phase, self.cfg.controller_type)
        if phase != self._phase and self._phase is not None:
            prev = phase_to_tag(self._phase, self.cfg.controller_type)
            if prev == "pick":
                self._held = True
            elif prev == "place":
                self._held = False
            self.extractor.complete_phase(prev)
        self._phase = phase
        return tag

    def step(self, state: dict[str, Any]) -> tuple[Any, bool, bool]:
        action, done, success = self.controller.step(state)
        tag = self._update_phase(_phase_name(self.controller))
        if action is None:
            return action, done, success
        shadow = self.extractor(state, held=self._held or tag in ("pour", "shake", "transport"))
        src, dst = self.extractor.source_id, self.extractor.target_id
        proposed = self.bridge.to_labguard(action, shadow, tag, src, dst if tag in ("pour", "place") else None)
        d = self.shield.filter(shadow, proposed)
        self.decisions.append(d)
        self.steps += 1
        if d.verdict == "pass":
            return action, done, success
        self.interventions += 1
        return self.bridge.to_articulation(d.action, action, shadow), done, success

    def reset(self) -> None:
        self.controller.reset()
        self.extractor.reset()
        self.shield.reset()
        self.decisions.clear()
        self._phase = None
        self._held = False

    def stats(self) -> dict[str, Any]:
        return {"steps": self.steps, "interventions": self.interventions,
                "intervention_rate": self.interventions / max(self.steps, 1),
                "logged": self.shield.logger.counts()}
