"""Compiled runtime monitor m_z = compile(z)  (Sec. 3.4, Eq. 5-6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from labguard.compiler.library import BaseCheck
from labguard.core.action import Action
from labguard.core.state import LabState
from labguard.ir.schema import LabGuardIR


@dataclass
class MonitorResult:
    rule_id: str | None
    monitor: str
    constraint_type: str
    severity: str
    intervention: str
    active: bool
    violated: bool | None  # None: inactive or predicate not applicable
    verdict: str  # pass | modify | block
    action: Action  # repaired action when verdict == modify, otherwise the input
    message: str = ""
    changes: dict[str, Any] = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        return bool(self.violated)


@dataclass
class CompiledMonitor:
    ir: LabGuardIR
    check: BaseCheck

    @property
    def rule_id(self) -> str | None:
        return self.ir.rule_id

    @property
    def scope(self) -> frozenset[str]:
        return frozenset(self.ir.action_scope)

    @property
    def severity(self) -> str:
        return self.ir.severity

    @property
    def intervention(self) -> str:
        return self.ir.intervention

    @property
    def theta(self) -> dict[str, Any]:
        return self.check.theta()

    def describe(self) -> str:
        s = ",".join(self.ir.substances) or "*"
        t = ",".join(self.ir.action_scope) or "*"
        return (f"{self.check.name}(S={{{s}}}, T={{{t}}}, p=`{self.check.predicate.canonical}`) "
                f"-> {self.intervention}/{self.severity}")

    # ----------------------------------------------------------- Eq. 5
    def in_scope(self, action: Action) -> bool:
        return not self.scope or action.tag in self.scope

    def is_active(self, state: LabState, action: Action) -> bool:
        return self.in_scope(action) and self.check.binds(state, action)

    def violated(self, state: LabState, action: Action) -> bool | None:
        """True/False if the monitor is active and its predicate applicable, else None."""
        if not self.is_active(state, action):
            return None
        ok = self.check.evaluate(state, action)
        return None if ok is None else not ok

    # ----------------------------------------------------------- Eq. 6
    def __call__(self, state: LabState, action: Action) -> MonitorResult:
        base = dict(rule_id=self.rule_id, monitor=self.check.name, constraint_type=self.ir.constraint_type,
                    severity=self.severity, intervention=self.intervention)
        if not self.is_active(state, action):
            return MonitorResult(**base, active=False, violated=None, verdict="pass", action=action)
        ok = self.check.evaluate(state, action)
        if ok is None or ok:
            return MonitorResult(**base, active=True, violated=None if ok is None else False,
                                 verdict="pass", action=action)
        msg = self.check.explain(state, action)
        if self.intervention == "log":
            return MonitorResult(**base, active=True, violated=True, verdict="pass", action=action,
                                 message="[logged] " + msg)
        if self.intervention == "modify":
            repaired = self.check.repair(state, action)
            if repaired is not None and self.check.evaluate(state, repaired) is not False:
                return MonitorResult(**base, active=True, violated=True, verdict="modify", action=repaired,
                                     message="[modified] " + msg, changes=action.diff(repaired))
            msg += " (no admissible repair; escalated to block)"
        return MonitorResult(**base, active=True, violated=True, verdict="block", action=action,
                             message="[blocked] " + msg)
