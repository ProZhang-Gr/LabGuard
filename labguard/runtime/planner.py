"""Plan-level checking ("Planning only" ablation, Sec. 5.3).

The PlanChecker verifies a skill-level plan *before* execution by rolling it
out on a copy of the state with a transition model and evaluating the
compiled monitors at every step.  It catches unsafe plan structure (wrong
order, incompatible transfers) but, by construction, cannot catch deviations
introduced by the executing policy at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from labguard.compiler.monitor import CompiledMonitor
from labguard.core.action import Action
from labguard.core.state import LabState
from labguard.ir.registry import Registry, default_registry
from labguard.runtime.safe_filter import SafeFilter, SemanticDecision

Transition = Callable[[LabState, Action], object]


def _default_transition(registry: Registry) -> Transition:
    from labguard.envs.lab_physics import apply_semantic

    return lambda s, a: apply_semantic(s, a, registry)


@dataclass
class PlanStepReport:
    index: int
    action: Action
    decision: SemanticDecision


@dataclass
class PlanReport:
    ok: bool
    steps: list[PlanStepReport] = field(default_factory=list)
    repaired_plan: list[Action] = field(default_factory=list)

    @property
    def violations(self) -> list[PlanStepReport]:
        return [s for s in self.steps if s.decision.verdict != "pass"]

    @property
    def first_block(self) -> int | None:
        return next((s.index for s in self.steps if s.decision.verdict == "block"), None)


class PlanChecker:
    def __init__(self, monitors: list[CompiledMonitor], transition: Transition | None = None,
                 registry: Registry | None = None) -> None:
        self.filter = SafeFilter(monitors)
        self.registry = registry or default_registry()
        self.transition = transition or _default_transition(self.registry)

    def check(self, state: LabState, plan: list[Action], stop_on_block: bool = False) -> PlanReport:
        sim = state.copy()
        report = PlanReport(ok=True)
        for k, action in enumerate(plan):
            d = self.filter(sim, action)
            report.steps.append(PlanStepReport(k, action, d))
            if d.verdict == "block":
                report.ok = False
                if stop_on_block:
                    break
                continue  # a blocked step is skipped in the rollout
            executed = d.action or action
            report.repaired_plan.append(executed)
            self.transition(sim, executed)
        return report
