"""Runtime safety layer pi_safe (Eq. 1):  a'_t = pi_safe(s_t, a_t) in {a_t, a~_t, empty}.

Per control step:
  1. scope activation of the compiled monitors (Eq. 5),
  2. SafeFilter semantic checks with typed repairs (Eq. 6-7),
  3. kinematic ActionFilter (Eq. 8-11) on the (possibly repaired) action,
  4. diagnostic risk score (Eq. 12) and structured logging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from labguard.compiler.monitor import CompiledMonitor, MonitorResult
from labguard.core.action import Action
from labguard.core.state import LabState
from labguard.runtime.kinematics import ActionFilter, FilterEvent
from labguard.runtime.logger import SafetyLogger
from labguard.runtime.risk import RiskScore, RiskScorer
from labguard.runtime.safe_filter import SafeFilter


@dataclass
class StepDecision:
    verdict: str  # pass | modify | block
    action: Action | None  # executed action (None when blocked)
    proposed: Action
    semantic: list[MonitorResult] = field(default_factory=list)
    filter_events: list[FilterEvent] = field(default_factory=list)
    risk: RiskScore | None = None
    latency_ms: float = 0.0
    messages: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.verdict == "block"

    @property
    def fired_rules(self) -> list[str]:
        return [r.rule_id or r.monitor for r in self.semantic if r.violated]


class RuntimeExecutor:
    def __init__(self, monitors: list[CompiledMonitor] | None = None,
                 action_filter: ActionFilter | None = None, risk_scorer: RiskScorer | None = None,
                 logger: SafetyLogger | None = None, derive_speed: bool = True) -> None:
        self.safe_filter = SafeFilter(monitors or []) if monitors is not None else None
        self.action_filter = action_filter
        self.risk_scorer = risk_scorer
        self.logger = logger or SafetyLogger(enabled=False)
        self.derive_speed = derive_speed

    @property
    def monitors(self) -> list[CompiledMonitor]:
        return self.safe_filter.monitors if self.safe_filter else []

    def _prepare(self, state: LabState, action: Action) -> Action:
        """Fill derived quantities (EE speed of a joint-space command)."""
        if not self.derive_speed or action.speed is not None:
            return action
        kin = self.action_filter.kin if self.action_filter else None
        if kin is None:
            return action
        dt = self.action_filter.config.dt  # type: ignore[union-attr]
        qdot = action.qdot
        if qdot is None and action.q_target is not None:
            qdot = (action.q_target - state.robot.q) / dt
        if qdot is None:
            return action
        out = action.copy()
        out.speed = kin.ee_speed(state.robot.q, qdot)
        out.meta["derived_speed"] = True
        return out

    def step(self, state: LabState, action: Action) -> StepDecision:
        t0 = time.perf_counter()
        proposed = self._prepare(state, action)
        verdict, current = "pass", proposed
        semantic: list[MonitorResult] = []
        messages: list[str] = []
        if self.safe_filter is not None and len(self.safe_filter):
            sd = self.safe_filter(state, proposed)
            semantic = sd.results
            messages += sd.messages
            if sd.verdict == "block":
                verdict, current = "block", None
            elif sd.verdict == "modify":
                verdict, current = "modify", sd.action
        events: list[FilterEvent] = []
        if current is not None and self.action_filter is not None:
            current, events = self.action_filter(state, current)
            if events:
                verdict = "modify"
                messages += [f"[kinematic] {e.kind}: {e.detail}" for e in events]
        risk = self.risk_scorer(state, proposed) if self.risk_scorer else None
        latency = (time.perf_counter() - t0) * 1e3
        decision = StepDecision(verdict, current, action, semantic, events, risk, latency, messages)
        self.logger.log(
            t=state.time, tag=action.tag, proposed=action.describe(),
            executed=current.describe() if current else None, verdict=verdict,
            fired=[{"rule_id": r.rule_id, "monitor": r.monitor, "verdict": r.verdict, "message": r.message}
                   for r in semantic if r.violated],
            kinematic=[e.kind for e in events], risk=risk.total if risk else None,
            high_risk=bool(risk and risk.high), latency_ms=latency,
        )
        return decision

    __call__ = step
