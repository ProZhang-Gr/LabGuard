"""ToyLabEnv: lightweight laboratory simulator used in place of LabUtopia.

The environment owns the true LabState and advances it with the outcome
oracle in `lab_physics`.  It knows nothing about the safety layer: hazards
are judged purely physically, so a configuration without LabGuard can be
scored with the same oracle as one with LabGuard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from labguard.core.action import Action
from labguard.core.franka import PandaKinematics
from labguard.core.state import LabState
from labguard.envs.lab_physics import Hazard, apply_action
from labguard.envs.tasks import TaskSpec, get_task
from labguard.ir.registry import Registry, default_registry

COLLISION_HAZARDS = frozenset({"collision_env", "collision_table"})


@dataclass
class EnvStep:
    hazards: list[Hazard] = field(default_factory=list)
    t: float = 0.0

    @property
    def fatal(self) -> bool:
        return any(h.fatal for h in self.hazards)


class ToyLabEnv:
    def __init__(self, registry: Registry | None = None, dt: float = 0.05,
                 kin: PandaKinematics | None = None) -> None:
        self.registry = registry or default_registry()
        self.dt = dt
        self.kin = kin or PandaKinematics()
        self.task: TaskSpec | None = None
        self.state: LabState | None = None
        self.context: dict = {}
        self.hazards: list[Hazard] = []
        self.steps = 0
        self.rng = np.random.default_rng(0)

    def reset(self, task: str | TaskSpec, seed: int = 0, no_ppe: bool = False) -> LabState:
        self.task = get_task(task) if isinstance(task, str) else task
        self.rng = np.random.default_rng(seed)
        self.state = self.task.setup(self.rng)
        if no_ppe:
            self.state.ppe = set()
        self.context = self.task.initial_context(self.state)
        self.hazards = []
        self.steps = 0
        return self.state

    def step(self, action: Action | None) -> EnvStep:
        """Execute one control step; `None` (blocked action) keeps the robot still."""
        assert self.state is not None, "call reset() first"
        self.steps += 1
        if action is None:
            self.state.robot.qdot = np.zeros(7)
            self.state.robot.speed = 0.0
            self.state.time += self.dt
            return EnvStep([], self.state.time)
        hz = apply_action(self.state, action, self.registry, self.dt, self.kin)
        self.hazards += hz
        return EnvStep(hz, self.state.time)

    # ------------------------------------------------------------- outcome
    def goal_reached(self) -> bool:
        assert self.task is not None and self.state is not None
        try:
            return bool(self.task.goal(self.state, self.context))
        except KeyError:
            return False

    @property
    def fatal(self) -> bool:
        return any(h.fatal for h in self.hazards)

    def success(self) -> bool:
        """Task completed without any fatal hazard."""
        return self.goal_reached() and not self.fatal

    def collision_events(self) -> int:
        """Number of collision onsets (consecutive colliding steps count once)."""
        times = sorted({round(h.t, 6) for h in self.hazards if h.kind in COLLISION_HAZARDS})
        events, last = 0, None
        for t in times:
            if last is None or t - last > self.dt * 1.5:
                events += 1
            last = t
        return events
