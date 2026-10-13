"""Episode runner and runtime metrics (Sec. 5.4-5.5, Appendix C).

  SR   fraction of episodes that complete the task (goal reached, no fatal hazard)
  IR   fraction of controller steps in which the safety layer blocks / modifies
  VR   safety violations per episode, either logged by the active assurance
       layer (L1-L4 runs) or obtained by rescoring the executed trajectory with
       an offline compiled-monitor checker (controlled pour ablation)
  CR   collision events per episode
  EL   episode length in control steps
  TCT  total cycle time = simulated time + safety-layer compute overhead
  Risk mean heuristic risk score (Eq. 12)

Violations are counted as *events*: a rule that stays violated over
consecutive control steps counts once.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from labguard.compiler.monitor import CompiledMonitor
from labguard.core.action import Action
from labguard.core.state import LabState
from labguard.envs.lab_physics import Hazard
from labguard.envs.policies import NoisyScriptedPolicy, PolicyConfig
from labguard.envs.tasks import TaskSpec, get_task
from labguard.envs.toy_lab import ToyLabEnv
from labguard.runtime.kinematics import ActionFilter
from labguard.runtime.risk import RiskScorer
from labguard.runtime.shield import SafetyShield


class OfflineChecker:
    """Compiled-monitor checker used to rescore executed trajectories."""

    def __init__(self, monitors: list[CompiledMonitor], action_filter: ActionFilter | None = None,
                 risk_scorer: RiskScorer | None = None) -> None:
        self.monitors = monitors
        self.action_filter = action_filter if action_filter is not None else ActionFilter()
        self.risk_scorer = risk_scorer if risk_scorer is not None else RiskScorer()

    @classmethod
    def from_shield(cls, shield: SafetyShield) -> "OfflineChecker":
        return cls(shield.monitors)

    def violations(self, state: LabState, action: Action) -> set[str]:
        out = {m.ir.rule_id or m.check.name for m in self.monitors
               if m.intervention != "log" and m.violated(state, action)}
        if self.action_filter is not None:
            out |= {f"kin:{e.kind}" for e in self.action_filter.violations(state, action)}
        return out

    def risk(self, state: LabState, action: Action) -> float:
        return self.risk_scorer(state, action).total


class _EventCounter:
    """Counts onsets of violation keys over a sequence of control steps."""

    def __init__(self) -> None:
        self.active: set[str] = set()
        self.count = 0
        self.by_key: dict[str, int] = {}

    def update(self, keys: set[str]) -> None:
        for k in keys - self.active:
            self.count += 1
            self.by_key[k] = self.by_key.get(k, 0) + 1
        self.active = set(keys)


@dataclass
class EpisodeResult:
    task: str
    level: str
    seed: int
    config: str
    success: bool
    goal: bool
    steps: int
    interventions: int
    blocks: int
    modifies: int
    logged_violations: int
    offline_violations: int | None
    collisions: int
    sim_time: float
    overhead_s: float
    mean_risk: float | None
    hazards: list[str] = field(default_factory=list)
    fatal_hazards: list[str] = field(default_factory=list)
    injected_errors: list[str] = field(default_factory=list)
    plan_interventions: int = 0
    violated_rules: dict[str, int] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def intervention_rate(self) -> float:
        return self.interventions / max(self.steps, 1)

    @property
    def tct(self) -> float:
        return self.sim_time + self.overhead_s

    def to_dict(self, with_trace: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not with_trace:
            d.pop("trace")
        d["intervention_rate"] = self.intervention_rate
        d["tct"] = self.tct
        return d


def _plan_phase(shield: SafetyShield, policy: NoisyScriptedPolicy, state: LabState, rounds: int) -> int:
    """Check the skill plan before execution; returns the number of plan-level interventions."""
    n = 0
    for r in range(rounds):
        report = shield.check_plan(state, policy.plan_actions(state))
        if report is None:
            return n
        if not report.violations:
            break
        n += len(report.violations)
        policy.apply_plan_report(report, state, final=(r == rounds - 1))
    return n


def run_episode(task: str | TaskSpec, shield: SafetyShield | None = None, seed: int = 0,
                policy: NoisyScriptedPolicy | None = None, policy_config: PolicyConfig | None = None,
                checker: OfflineChecker | None = None, env: ToyLabEnv | None = None, config_name: str = "",
                no_ppe: bool = False, plan_rounds: int = 2, record: bool = False) -> EpisodeResult:
    spec = get_task(task) if isinstance(task, str) else task
    env = env or ToyLabEnv()
    state = env.reset(spec, seed=seed, no_ppe=no_ppe)
    policy = policy or NoisyScriptedPolicy(policy_config)
    policy.reset(spec, state, np.random.default_rng(10_000 + seed))
    if shield is not None:
        shield.reset()

    plan_interventions = 0
    overhead = 0.0
    if shield is not None and shield.plan_checker is not None:
        t0 = time.perf_counter()
        plan_interventions = _plan_phase(shield, policy, state, plan_rounds)
        overhead += time.perf_counter() - t0

    logged, offline = _EventCounter(), _EventCounter()
    interventions = blocks = modifies = 0
    risks: list[float] = []
    trace: list[dict[str, Any]] = []
    for _ in range(spec.max_steps):
        proposed = policy.act(env.state)
        if proposed is None:
            break
        executed: Action | None = proposed
        verdict = "pass"
        if shield is not None:
            t0 = time.perf_counter()
            d = shield.filter(env.state, proposed)
            overhead += time.perf_counter() - t0
            executed, verdict = d.action, d.verdict
            logged.update({r.rule_id or r.monitor for r in d.semantic if r.violated}
                          | {f"kin:{e.kind}" for e in d.filter_events})
            if d.risk is not None:
                risks.append(d.risk.total)
            policy.notify(proposed, verdict, env.state)
        if verdict != "pass":
            interventions += 1
            blocks += verdict == "block"
            modifies += verdict == "modify"
        if checker is not None:
            offline.update(checker.violations(env.state, executed) if executed is not None else set())
            if shield is None or shield.executor.risk_scorer is None:
                risks.append(checker.risk(env.state, proposed))
        res = env.step(executed)
        if record:
            trace.append({"t": res.t, "proposed": proposed.describe(), "verdict": verdict,
                          "executed": executed.describe() if executed else None,
                          "hazards": [h.kind for h in res.hazards]})

    hazards: list[Hazard] = env.hazards
    return EpisodeResult(
        task=spec.name, level=spec.level, seed=seed, config=config_name,
        success=env.success(), goal=env.goal_reached(), steps=env.steps,
        interventions=interventions, blocks=blocks, modifies=modifies,
        logged_violations=logged.count,
        offline_violations=offline.count if checker is not None else None,
        collisions=env.collision_events(), sim_time=float(env.state.time), overhead_s=overhead,
        mean_risk=float(np.mean(risks)) if risks else None,
        hazards=[h.kind for h in hazards], fatal_hazards=[h.kind for h in hazards if h.fatal],
        injected_errors=list(policy.injected), plan_interventions=plan_interventions,
        violated_rules=dict(offline.by_key if checker is not None else logged.by_key), trace=trace,
    )


def summarize(results: Iterable[EpisodeResult]) -> dict[str, Any]:
    rs = list(results)
    if not rs:
        return {}

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    offline = [r.offline_violations for r in rs if r.offline_violations is not None]
    total_steps = sum(r.steps for r in rs)
    return {
        "episodes": len(rs),
        "SR": mean([float(r.success) for r in rs]),
        "IR": sum(r.interventions for r in rs) / max(total_steps, 1),
        # plan-level corrections replace proposed actions before execution (Appendix C: "blocks,
        # modifies, or replaces"); reported separately so that L1-L4 IR stays step-level
        "IR_with_plan": sum(r.interventions + r.plan_interventions for r in rs) / max(total_steps, 1),
        "VR_logged": mean([r.logged_violations for r in rs]),
        "VR_offline": mean(offline) if offline else None,
        "Risk": mean([r.mean_risk for r in rs if r.mean_risk is not None]),
        "CR": mean([r.collisions for r in rs]),
        "EL": mean([r.steps for r in rs]),
        "TCT": mean([r.tct for r in rs]),
        "unsafe_event_rate": mean([float(bool(r.fatal_hazards)) for r in rs]),
        "plan_interventions": mean([r.plan_interventions for r in rs]),
    }


def seed_std(results: Iterable[EpisodeResult], key: str = "success") -> float:
    """Std over seeds of the per-seed mean (Fig. 3 error bars)."""
    by_seed: dict[int, list[float]] = {}
    for r in results:
        by_seed.setdefault(r.seed // 1000, []).append(float(getattr(r, key)))
    return float(np.std([np.mean(v) for v in by_seed.values()])) if by_seed else 0.0
