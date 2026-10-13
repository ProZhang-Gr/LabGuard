"""SafetyShield: user-facing facade that assembles the LabGuard runtime.

Modules (each can be switched off for the ablation in Sec. 5.3):
  semantic   compiled rule monitors (SafeFilter)
  kinematic  ActionFilter (workspace clamp, velocity clamp, joint barriers)
  planning   PlanChecker on the skill-level plan before execution
  risk       diagnostic risk score
  logging    structured per-step log
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from labguard.compiler.compiler import CompileError, Compiler
from labguard.compiler.monitor import CompiledMonitor
from labguard.core.action import Action
from labguard.core.state import LabState
from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import LabGuardIR
from labguard.runtime.executor import RuntimeExecutor, StepDecision
from labguard.runtime.kinematics import ActionFilter, ActionFilterConfig
from labguard.runtime.logger import SafetyLogger
from labguard.runtime.planner import PlanChecker, PlanReport
from labguard.runtime.risk import RISK_WEIGHTS, RiskScorer
from labguard.runtime.rules import RuntimeRule, gold_irs, ground_rules, load_rules

if TYPE_CHECKING:
    from labguard.grounder.base import Grounder


@dataclass
class ShieldConfig:
    semantic: bool = True
    kinematic: bool = True
    planning: bool = False
    risk: bool = True
    logging: bool = True
    filter: ActionFilterConfig = field(default_factory=ActionFilterConfig)
    risk_weights: tuple[float, float, float, float] = RISK_WEIGHTS  # w_v, w_p, w_c, w_u (Eq. 12)
    risk_tau: float = 0.6

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ShieldConfig":
        d = dict(d or {})
        filt = ActionFilterConfig(**d.pop("filter", {}))
        if "risk_weights" in d:
            d["risk_weights"] = tuple(d["risk_weights"])
        return cls(filter=filt, **d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetyShield:
    def __init__(self, monitors: list[CompiledMonitor], config: ShieldConfig | None = None,
                 registry: Registry | None = None, compile_errors: list[CompileError] | None = None) -> None:
        self.config = config or ShieldConfig()
        self.registry = registry or default_registry()
        self.monitors = monitors
        self.compile_errors = compile_errors or []
        cfg = self.config
        self.logger = SafetyLogger(enabled=cfg.logging)
        w_v, w_p, w_c, w_u = cfg.risk_weights
        self.executor = RuntimeExecutor(
            monitors=monitors if cfg.semantic else None,
            action_filter=ActionFilter(cfg.filter) if cfg.kinematic else None,
            risk_scorer=RiskScorer(w_v, w_p, w_c, w_u, cfg.risk_tau, dt=cfg.filter.dt, registry=self.registry)
            if cfg.risk else None,
            logger=self.logger,
        )
        self.plan_checker = PlanChecker(monitors, registry=self.registry) if cfg.planning else None

    # ------------------------------------------------------------- builders
    @classmethod
    def from_irs(cls, irs: list[LabGuardIR], config: ShieldConfig | None = None,
                 registry: Registry | None = None) -> "SafetyShield":
        registry = registry or default_registry()
        monitors, errors = Compiler(registry).compile_many(irs)
        return cls(monitors, config, registry, errors)

    @classmethod
    def from_rules(cls, rules: list[RuntimeRule] | str | None = None, grounder: "Grounder | None" = None,
                   config: ShieldConfig | None = None, registry: Registry | None = None) -> "SafetyShield":
        """Ground rule text with `grounder` (or use the gold IR when grounder is None)."""
        if rules is None or isinstance(rules, str):
            rules = load_rules(rules)
        irs = gold_irs(rules) if grounder is None else ground_rules(rules, grounder)
        return cls.from_irs(irs, config, registry)

    # -------------------------------------------------------------- runtime
    def filter(self, state: LabState, action: Action) -> StepDecision:
        return self.executor.step(state, action)

    __call__ = filter

    def check_plan(self, state: LabState, plan: list[Action]) -> PlanReport | None:
        if self.plan_checker is None:
            return None
        return self.plan_checker.check(state, plan)

    def reset(self) -> None:
        self.logger.clear()

    def describe(self) -> str:
        lines = [m.describe() for m in self.monitors]
        lines += [f"REJECTED {e.ir.rule_id if e.ir else '?'}: {e}" for e in self.compile_errors]
        return "\n".join(lines)
