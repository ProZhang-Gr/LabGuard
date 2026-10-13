"""Scripted state/action scenarios for the replay diagnostics (Sec. 5.3).

`ScenarioSampler` turns a gold LabGuard-IR into concrete (s_t, a_t) pairs:
it randomizes the toy-lab scene (PPE, completed steps, equipment and
glassware flags, positions), picks a task tag from the IR scope and the
substances from S, and places every numeric quantity referenced by the
predicate near its threshold (value x U(0.5, 1.5)).  The *gold* monitor then
labels each sample as violating or compliant; rejection sampling collects one
of each per rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from labguard.compiler.monitor import CompiledMonitor
from labguard.core.action import Action
from labguard.core.state import Container, LabState
from labguard.envs.scene import default_scene
from labguard.ir.predicate import Predicate
from labguard.ir.registry import Registry, default_registry
from labguard.vocab import PLACEMENT_TAGS, TASK_TAGS, TRANSFER_TAGS

_DIST_PATH = re.compile(r"distance\('([A-Za-z_0-9]+)'\)")
_WORKSPACE_LO = np.array([0.22, -0.40])
_WORKSPACE_HI = np.array([0.70, 0.40])
_CONTAINER_ATTRS = ("temperature", "open_time", "fill_ratio", "volume", "capacity")


@dataclass
class Scenario:
    state: LabState
    action: Action
    violating: bool


class ScenarioSampler:
    def __init__(self, registry: Registry | None = None, max_tries: int = 400) -> None:
        self.registry = registry or default_registry()
        self.max_tries = max_tries

    # ------------------------------------------------------------------ helpers
    def _xy(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([*rng.uniform(_WORKSPACE_LO, _WORKSPACE_HI), 0.0])

    def _in_zone(self, zone: str, rng: np.random.Generator) -> np.ndarray:
        lo, hi = (np.asarray(b, dtype=float) for b in self.registry.zones[zone])
        p = rng.uniform(lo + 0.005, hi - 0.005)
        p[2] = 0.0
        return p

    def _mixture(self, pool: list[str], rng: np.random.Generator) -> dict[str, float]:
        reg = self.registry
        if pool and rng.random() < 0.8:
            sub = str(rng.choice(pool))
        else:
            sub = str(rng.choice(reg.chemical_ids))
        return {sub: float(rng.uniform(20.0, 150.0))}

    def _container(self, cid: str, contents: dict[str, float], rng: np.random.Generator,
                   kind: str | None = None) -> Container:
        kind = kind or str(rng.choice(["beaker", "flask", "bottle", "test_tube"]))
        cap = {"beaker": 250.0, "flask": 250.0, "bottle": 500.0, "test_tube": 50.0}.get(kind, 250.0)
        c = Container(cid, kind, contents, cap, temperature=float(rng.uniform(15.0, 80.0)),
                      has_lid=kind != "beaker" or rng.random() < 0.2, lid_open=bool(rng.random() < 0.6),
                      open_time=float(rng.uniform(0.0, 200.0)), position=self._xy(rng),
                      intact=bool(rng.random() < 0.8), clean=bool(rng.random() < 0.7), dry=bool(rng.random() < 0.7))
        if c.volume > c.capacity:
            scale = 0.9 * c.capacity / c.volume
            c.contents = {k: v * scale for k, v in c.contents.items()}
        return c

    @staticmethod
    def _set_volume(c: Container, volume: float) -> None:
        volume = max(0.0, volume)
        if not c.contents:
            c.contents = {"H2O": volume}
            return
        scale = volume / c.volume if c.volume > 0 else 0.0
        c.contents = {k: v * scale for k, v in c.contents.items()}

    # ------------------------------------------------------------------ sampling
    def sample(self, ir: Any, predicate: Predicate, rng: np.random.Generator) -> tuple[LabState, Action]:
        reg = self.registry
        s = default_scene(rng, jitter=0.02)
        s.ppe = {p for p in reg.ppe_items if rng.random() < 0.6}
        s.done_steps = [st for st in sorted(reg.step_names()) if rng.random() < 0.25]
        for e in s.equipment.values():
            for f in sorted(set(e.flags) | set(reg.equipment_flags.get(e.kind, ()))):
                e.flags[f] = bool(rng.random() < 0.65)
            e.setpoint = float(rng.uniform(20.0, 250.0))
        s.robot.speed = float(rng.uniform(0.0, 0.6))
        s.robot.gripper_force = float(rng.uniform(0.0, 45.0))

        tag = str(rng.choice(ir.action_scope)) if ir.action_scope else str(rng.choice(TASK_TAGS))
        literal_chems = [a for _, args in predicate.calls for a in args
                         if isinstance(a, str) and reg.is_chemical(reg.resolve(a, fuzzy=False) or "")]
        pool = [x for x in ir.substances if reg.is_chemical(x)] + [reg.resolve(a, fuzzy=False) for a in literal_chems]
        pool = [p for p in pool if p]

        src = self._container("vessel_src", self._mixture(pool, rng), rng)
        if rng.random() < 0.3:  # occasional mixture of two chemicals in the source
            src.add(self._mixture(pool, rng))
        s.containers[src.id] = src
        a = Action(tag, src.id)
        dst: Container | None = None

        if tag in TRANSFER_TAGS:
            if tag == "dispose":
                label = str(rng.choice(reg.waste_streams))
                dst = Container("waste_x", "waste_container", {}, 2000.0, label=label if rng.random() < 0.85 else None,
                                position=self._xy(rng), lid_open=bool(rng.random() < 0.8))
                if rng.random() < 0.7:
                    dst.add(self._mixture(pool, rng))
                    self._set_volume(dst, float(rng.uniform(0.05, 1.0)) * dst.capacity)
            else:
                dst = self._container("vessel_dst", self._mixture(pool, rng) if rng.random() < 0.6 else {}, rng,
                                      kind=str(rng.choice(["beaker", "flask"])))
            s.containers[dst.id] = dst
            a.dst = dst.id
            a.volume = float(rng.uniform(5.0, 150.0))
            a.pour_rate = float(rng.uniform(5.0, 50.0))
            a.tilt_angle = float(rng.uniform(10.0, 90.0)) if rng.random() < 0.5 else None
        elif tag in ("heat", "cool"):
            a.dst = "hot_plate_1"
            a.target_temperature = float(rng.uniform(20.0, 200.0)) if tag == "heat" else float(rng.uniform(0, 20))
        elif tag in PLACEMENT_TAGS | {"transport", "move"}:
            if rng.random() < 0.5:
                a.dst = str(rng.choice(sorted(reg.zones)))
            else:
                a.target_pos = self._xy(rng) + np.array([0.0, 0.0, 0.05])
            a.speed = float(rng.uniform(0.05, 0.6))
            if tag == "move":
                a.src = None
        elif tag in ("press", "centrifuge"):
            a.dst = str(rng.choice(sorted(s.equipment)))
        if rng.random() < 0.5:
            a.grip_force = float(rng.uniform(5.0, 45.0))
        if rng.random() < 0.5:
            a.duration = float(rng.uniform(1.0, 300.0))
        if a.speed is None and rng.random() < 0.4:
            a.speed = float(rng.uniform(0.05, 0.6))

        self._target_predicate(s, a, src, dst, predicate, rng)
        return s, a

    def _op_entity(self, s: LabState, a: Action) -> Any:
        """Entity whose position is the operation position (see core.context.operation_position)."""
        if a.target_pos is not None:
            return None
        if a.dst is not None:
            return s.containers.get(a.dst) or s.equipment.get(a.dst)
        return s.containers.get(a.src) if a.src else None

    def _set_op_position(self, s: LabState, a: Action, p: np.ndarray) -> None:
        if a.target_pos is not None or (a.dst is not None and s.is_zone(a.dst)):
            a.dst = None if a.dst is not None and s.is_zone(a.dst) else a.dst
            a.target_pos = p + np.array([0.0, 0.0, 0.05])
            return
        ent = self._op_entity(s, a)
        if ent is not None:
            ent.position = p.copy()
        else:
            a.target_pos = p + np.array([0.0, 0.0, 0.05])

    def _outside(self, s: LabState, zone: str, rng: np.random.Generator) -> np.ndarray:
        for _ in range(50):
            p = np.array([*rng.uniform([0.0, -0.75], [0.95, 0.75]), 0.0])
            if not s.in_zone(p, zone):
                return p
        return p

    def _add_incompatible(self, s: LabState, a: Action, src: Container, rng: np.random.Generator) -> None:
        """Put a chemical incompatible with the source contents at the destination."""
        partners = sorted({y for x in src.substances for y in self.registry.inc(x)})
        if not partners:
            return
        mix = {str(rng.choice(partners)): float(rng.uniform(10.0, 60.0))}
        dst = s.containers.get(a.dst) if a.dst else None
        if dst is not None:
            dst.add(mix)
        elif a.dst is not None and s.is_zone(a.dst):
            s.containers["stored_x"] = Container("stored_x", "bottle", mix, 500.0, lid_open=False,
                                                 position=self._in_zone(a.dst, rng))

    def _target_predicate(self, s: LabState, a: Action, src: Container, dst: Container | None,
                          predicate: Predicate, rng: np.random.Generator) -> None:
        reg = self.registry
        # spatial calls first (they may move the operation point)
        for fn, args in predicate.calls:
            if fn == "in_zone" and args and args[0] in s.zones:
                r = rng.random()
                if r < 0.4:
                    self._set_op_position(s, a, self._in_zone(str(args[0]), rng))
                elif r < 0.7:
                    self._set_op_position(s, a, self._outside(s, str(args[0]), rng))
            elif fn == "contains" and len(args) == 2 and rng.random() < 0.6:
                sub = reg.resolve(str(args[1]), fuzzy=False) or str(args[1])
                view = dst if args[0] == "dst" and dst is not None else src
                view.add({sub: float(rng.uniform(10.0, 60.0))})
            elif fn == "incompatible" and rng.random() < 0.5:
                self._add_incompatible(s, a, src, rng)
        for c in predicate.comparisons:
            val = c.value
            if isinstance(val, bool):
                continue
            if isinstance(val, str):
                if c.path == "dst.label" and dst is not None and rng.random() < 0.5:
                    dst.label = val
                elif c.path == "src.label" and rng.random() < 0.5:
                    src.label = val
                continue
            if not isinstance(val, (int, float)):
                continue
            v = float(val) * rng.uniform(0.5, 1.5) if val != 0 else float(rng.uniform(-1.0, 1.0))
            self._set_path(s, a, src, dst, c.path, v, rng)

    def _set_path(self, s: LabState, a: Action, src: Container, dst: Container | None, path: str, v: float,
                  rng: np.random.Generator) -> None:
        m = _DIST_PATH.fullmatch(path)
        if m:
            kind = self.registry.resolve(m.group(1)) or m.group(1)
            op = self._op_point(s, a)
            ang = rng.uniform(0, 2 * np.pi)
            for e in s.equipment_of_kind(kind):
                e.position = op + np.array([np.cos(ang), np.sin(ang), 0.0]) * max(v, 0.0)
                e.position[2] = 0.0
            return
        root, _, attr = path.partition(".")
        if root == "action":
            if attr == "fill_after":
                if dst is None:
                    return
                v = max(v, 0.01)
                a.volume = float(rng.uniform(0.1, 0.6)) * v * dst.capacity
                self._set_volume(dst, v * dst.capacity - a.volume)
            elif hasattr(a, attr):
                setattr(a, attr, v)
        elif root in ("src", "dst"):
            c = src if root == "src" else dst
            if c is None or attr not in _CONTAINER_ATTRS:
                return
            if attr == "fill_ratio":
                self._set_volume(c, max(v, 0.0) * c.capacity)
            elif attr == "volume":
                self._set_volume(c, v)
            else:
                setattr(c, attr, v)
        elif root == "robot" and attr in ("speed", "gripper_force"):
            setattr(s.robot, attr, v)
        elif root == "env" and attr == "ambient_temperature":
            s.ambient_temperature = v

    @staticmethod
    def _op_point(s: LabState, a: Action) -> np.ndarray:
        from labguard.core.context import operation_position

        return np.asarray(operation_position(s, a), dtype=float).copy()

    # ------------------------------------------------------------------ labelled pairs
    def find_cases(self, monitor: CompiledMonitor, rng: np.random.Generator,
                   want: tuple[bool, ...] = (True, False)) -> dict[bool, Scenario]:
        """Rejection-sample until one scenario per requested label is found (gold monitor labels)."""
        found: dict[bool, Scenario] = {}
        pred = monitor.check.predicate
        for _ in range(self.max_tries):
            state, action = self.sample(monitor.ir, pred, rng)
            v = monitor.violated(state, action)
            if v is None or v in found or v not in want:
                continue
            found[v] = Scenario(state, action, v)
            if len(found) == len(want):
                break
        return found
