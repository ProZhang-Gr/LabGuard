"""SafeFilter: composes the compiled semantic monitors (Eq. 1, 6).

Composition rule (verdict precedence block > modify > pass):
  1. evaluate every active monitor on the proposed action;
  2. any `block` verdict blocks the action;
  3. `modify` repairs are applied sequentially (each monitor sees the action
     already repaired by the previous ones);
  4. the final repaired action is re-checked against *all* monitors; if any
     non-log monitor is still violated the action is blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from labguard.compiler.monitor import CompiledMonitor, MonitorResult
from labguard.core.action import Action
from labguard.core.state import LabState
from labguard.vocab import SEVERITY_RANK


@dataclass
class SemanticDecision:
    verdict: str  # pass | modify | block
    action: Action | None  # None when blocked
    results: list[MonitorResult] = field(default_factory=list)
    recheck_failures: list[MonitorResult] = field(default_factory=list)

    @property
    def fired(self) -> list[MonitorResult]:
        return [r for r in self.results if r.violated]

    @property
    def messages(self) -> list[str]:
        return [r.message for r in self.results if r.message] + [r.message for r in self.recheck_failures]


class SafeFilter:
    def __init__(self, monitors: list[CompiledMonitor]) -> None:
        # high-severity monitors first so that their verdict/repair takes precedence
        self.monitors = sorted(monitors, key=lambda m: -SEVERITY_RANK.get(m.severity, 0))

    def __len__(self) -> int:
        return len(self.monitors)

    def __call__(self, state: LabState, action: Action) -> SemanticDecision:
        results: list[MonitorResult] = []
        current = action
        blocked = False
        for m in self.monitors:
            r = m(state, current)
            results.append(r)
            if r.verdict == "block":
                blocked = True
            elif r.verdict == "modify" and not blocked:
                current = r.action
        if blocked:
            return SemanticDecision("block", None, results)
        if current is action:
            return SemanticDecision("pass", action, results)
        failures = []
        for m in self.monitors:
            if m.intervention != "log" and m.violated(state, current):
                r = m(state, current)
                r.message = "[re-check] " + (r.message or "repaired action still violates rule")
                failures.append(r)
        if failures:
            return SemanticDecision("block", None, results, failures)
        return SemanticDecision("modify", current, results)
