"""Task suite mirroring the LabUtopia levels used in the paper (Sec. 5.1).

  L1 atomic skills     pick, place, shake, stir, open, close, pour
  L2 composite tasks   HeatLiquid, PourLiquid, ShakeBeaker, TransportBeaker
  L3 generalisation    press, HeatLiquidOOD (ethanol + moved hot plate)
  L4 long horizon      CleanBeaker

Each task provides an initial scene, a skill-level plan for the scripted
policy and a goal predicate.  Task success = goal reached with no fatal hazard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from labguard.core.state import Container, LabState
from labguard.envs.scene import CARRY_HEIGHT, above, default_scene, put_in_gripper


@dataclass
class Skill:
    kind: str  # pick place transport move pour heat shake stir open close press dispose clean inspect
    src: str | None = None
    dst: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Skill":
        return Skill(self.kind, self.src, self.dst, dict(self.params))


@dataclass
class TaskSpec:
    name: str
    level: str
    setup: Callable[[np.random.Generator], LabState]
    plan: Callable[[LabState], list[Skill]]
    goal: Callable[[LabState, dict[str, Any]], bool]
    max_steps: int = 1500
    description: str = ""

    def initial_context(self, state: LabState) -> dict[str, Any]:
        return {"volumes": {c.id: c.volume for c in state.containers.values()},
                "positions": {c.id: c.position.copy() for c in state.containers.values()}}


def _held(state: LabState, cid: str) -> bool:
    return state.robot.holding == cid


def _in_zone(state: LabState, cid: str, zone: str) -> bool:
    return state.in_zone(state.containers[cid].position, zone, margin=0.02)


def _kept(state: LabState, ctx: dict, cid: str, frac: float = 0.9) -> bool:
    return state.containers[cid].volume >= frac * ctx["volumes"][cid] - 1e-6


# ---------------------------------------------------------------------- L1
def _setup_default(rng: np.random.Generator) -> LabState:
    return default_scene(rng, jitter=0.01)


def _setup_holding(cid: str, over: str, height: float = CARRY_HEIGHT):
    def setup(rng: np.random.Generator) -> LabState:
        s = default_scene(rng, jitter=0.01)
        put_in_gripper(s, cid, above(s, over, height))
        return s
    return setup


def _setup_bottle_open(rng: np.random.Generator) -> LabState:
    s = default_scene(rng, jitter=0.01)
    s.containers["bottle_hcl"].lid_open = True
    return s


TASKS: dict[str, TaskSpec] = {}


def _register(t: TaskSpec) -> None:
    TASKS[t.name] = t


_register(TaskSpec(
    "pick", "L1", _setup_default,
    lambda s: [Skill("pick", "beaker_water")],
    lambda s, c: _held(s, "beaker_water") and s.robot.ee_pos[2] > 0.12 and _kept(s, c, "beaker_water"),
    description="Pick up the water beaker."))
_register(TaskSpec(
    "place", "L1", _setup_holding("beaker_water", "beaker_water"),
    lambda s: [Skill("transport", "beaker_water", "general_shelf"), Skill("place", "beaker_water", "general_shelf")],
    lambda s, c: (not _held(s, "beaker_water")) and _in_zone(s, "beaker_water", "general_shelf")
    and _kept(s, c, "beaker_water"),
    description="Place the held beaker on the shelf."))
_register(TaskSpec(
    "shake", "L1", _setup_holding("flask_ethanol", "flask_ethanol"),
    lambda s: [Skill("close", "flask_ethanol"), Skill("shake", "flask_ethanol")],
    lambda s, c: "shake" in s.done_steps and _kept(s, c, "flask_ethanol", 0.99),
    description="Shake the held flask."))
_register(TaskSpec(
    "stir", "L1", _setup_default,
    lambda s: [Skill("stir", "beaker_water")],
    lambda s, c: "stir" in s.done_steps,
    description="Stir the beaker."))
_register(TaskSpec(
    "open", "L1", _setup_default,
    lambda s: [Skill("open", "bottle_hcl")],
    lambda s, c: s.containers["bottle_hcl"].lid_open,
    description="Open the reagent bottle."))
_register(TaskSpec(
    "close", "L1", _setup_bottle_open,
    lambda s: [Skill("close", "bottle_hcl")],
    lambda s, c: not s.containers["bottle_hcl"].lid_open,
    description="Close the reagent bottle."))
_register(TaskSpec(
    "pour", "L1", _setup_holding("beaker_water", "beaker_target", 0.15),
    lambda s: [Skill("pour", "beaker_water", "beaker_target", {"volume": 60.0})],
    lambda s, c: s.containers["beaker_target"].volume >= 30.0 - 1e-6,
    description="Pour water from the held beaker into the target beaker."))


# ---------------------------------------------------------------------- L2
def _plan_heat(cid: str, temperature: float):
    def plan(s: LabState) -> list[Skill]:
        return [
            Skill("pick", cid),
            Skill("transport", cid, "hot_plate_1"),
            Skill("place", cid, "hot_plate_1"),
            Skill("heat", cid, "hot_plate_1", {"target_temperature": temperature}),
        ]
    return plan


_register(TaskSpec(
    "HeatLiquid", "L2", _setup_default, _plan_heat("beaker_water", 80.0),
    lambda s, c: s.containers["beaker_water"].temperature >= 60.0 and _kept(s, c, "beaker_water"),
    max_steps=2000, description="Move the water beaker onto the hot plate and heat it."))


def _plan_pour_liquid(s: LabState) -> list[Skill]:
    home = s.containers["bottle_hcl"].position.copy()
    return [
        Skill("open", "bottle_hcl"),
        Skill("pick", "bottle_hcl"),
        Skill("transport", "bottle_hcl", "beaker_target"),
        Skill("pour", "bottle_hcl", "beaker_target", {"volume": 50.0}),
        Skill("transport", "bottle_hcl", None, {"goal": home + np.array([0, 0, CARRY_HEIGHT])}),
        Skill("place", "bottle_hcl", None, {"goal": home}),
    ]


_register(TaskSpec(
    "PourLiquid", "L2", _setup_default, _plan_pour_liquid,
    lambda s, c: s.containers["beaker_target"].contents.get("HCl", 0.0) >= 25.0 - 1e-6
    and not _held(s, "bottle_hcl"),
    max_steps=2500, description="Open the HCl bottle and pour 50 mL into the target beaker."))


def _plan_shake_beaker(s: LabState) -> list[Skill]:
    home = s.containers["flask_ethanol"].position.copy()
    return [
        Skill("pick", "flask_ethanol"),
        Skill("close", "flask_ethanol"),
        Skill("shake", "flask_ethanol"),
        Skill("place", "flask_ethanol", None, {"goal": home}),
    ]


_register(TaskSpec(
    "ShakeBeaker", "L2", _setup_default, _plan_shake_beaker,
    lambda s, c: "shake" in s.done_steps and _kept(s, c, "flask_ethanol", 0.99)
    and not _held(s, "flask_ethanol"),
    max_steps=2000, description="Pick up the flask, close it, shake it and put it back."))
_register(TaskSpec(
    "TransportBeaker", "L2", _setup_default,
    lambda s: [Skill("pick", "beaker_water"), Skill("transport", "beaker_water", "sink"),
               Skill("place", "beaker_water", "sink")],
    lambda s, c: (not _held(s, "beaker_water")) and _in_zone(s, "beaker_water", "sink")
    and _kept(s, c, "beaker_water"),
    max_steps=2000, description="Carry the water beaker to the sink area."))


# ---------------------------------------------------------------------- L3
def _plan_press(s: LabState) -> list[Skill]:
    return [Skill("move", None, "centrifuge_1", {"height": 0.15}), Skill("press", None, "centrifuge_1")]


_register(TaskSpec(
    "press", "L3", _setup_default, _plan_press,
    lambda s, c: s.equipment["centrifuge_1"].flags.get("on", False),
    description="Press the start button of the centrifuge."))


def _setup_heat_ood(rng: np.random.Generator) -> LabState:
    s = default_scene(rng, jitter=0.02)
    s.equipment["hot_plate_1"].position = np.array([0.55, 0.30, 0.0])
    s.containers["flask_ethanol"].position = np.array([0.40, -0.05, 0.0])
    return s


_register(TaskSpec(
    "HeatLiquidOOD", "L3", _setup_heat_ood, _plan_heat("flask_ethanol", 75.0),
    lambda s, c: s.containers["flask_ethanol"].temperature >= 50.0 and _kept(s, c, "flask_ethanol"),
    max_steps=2000,
    description="Out-of-distribution HeatLiquid: ethanol instead of water, moved hot plate."))


# ---------------------------------------------------------------------- L4
def _setup_clean(rng: np.random.Generator) -> LabState:
    s = default_scene(rng, jitter=0.01)
    s.containers["beaker_dirty"] = Container(
        "beaker_dirty", "beaker", {"CHCl3": 30.0}, 250.0, has_lid=True, lid_open=True,
        position=np.array([0.40, -0.02, 0.0]))
    return s


def _plan_clean(s: LabState) -> list[Skill]:
    return [
        Skill("pick", "beaker_dirty"),
        Skill("transport", "beaker_dirty", "waste_halogenated"),
        Skill("dispose", "beaker_dirty", "waste_halogenated"),
        Skill("transport", "beaker_dirty", "sink"),
        Skill("clean", "beaker_dirty"),
        Skill("place", "beaker_dirty", "sink"),
    ]


_register(TaskSpec(
    "CleanBeaker", "L4", _setup_clean, _plan_clean,
    lambda s, c: s.containers["beaker_dirty"].volume <= 1e-6 and s.containers["beaker_dirty"].clean
    and s.containers["waste_halogenated"].contents.get("CHCl3", 0.0) > 0 and not _held(s, "beaker_dirty"),
    max_steps=3000, description="Empty the chloroform residue into waste, clean the beaker, leave it in the sink."))

# ------------------------------------------------- controlled pour (ablation)
def _setup_pour_stress(rng: np.random.Generator) -> LabState:
    s = default_scene(rng, jitter=0.015)
    s.containers["beaker_target"].contents = {"H2O": 120.0}
    s.containers["beaker_bleach"].position = s.containers["beaker_target"].position + np.array([0.09, 0.0, 0.0])
    return s


_register(TaskSpec(
    "PourStress", "ablation", _setup_pour_stress, _plan_pour_liquid,
    lambda s, c: s.containers["beaker_target"].contents.get("HCl", 0.0) >= 25.0 - 1e-6
    and s.containers["beaker_target"].volume <= s.containers["beaker_target"].capacity
    and not _held(s, "bottle_hcl"),
    max_steps=2500,
    description="Controlled adversarial pour scenario of the module ablation (Sec. 5.5): partly filled "
                "receiver, bleach beaker next to it."))

LEVELS: dict[str, list[str]] = {
    "L1": ["pick", "place", "shake", "stir", "open", "close", "pour"],
    "L2": ["HeatLiquid", "PourLiquid", "ShakeBeaker", "TransportBeaker"],
    "L3": ["press", "HeatLiquidOOD"],
    "L4": ["CleanBeaker"],
}


def get_task(name: str) -> TaskSpec:
    return TASKS[name]
