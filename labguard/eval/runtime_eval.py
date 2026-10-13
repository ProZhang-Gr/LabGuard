"""Runtime monitor execution (Sec. 5.4, Fig. 3, Table 9) and module ablation (Sec. 5.5, Table 5).

Both protocols run the weak `NoisyScriptedPolicy` (ACT stand-in) in
`ToyLabEnv` with and without the safety layer.  Episodes of seed s use
episode seeds s*1000 + k so that `seed_std` can recover per-seed means.

  run_runtime   L1-L4 tasks, Base vs Full LabGuard; SR mean±std over seeds,
                IR, assurance-layer-logged VR, unsafe-event rate
  run_ablation  controlled PourStress scenario, one ShieldConfig per module;
                VR is rescored offline by an OfflineChecker over the gold
                monitors (outcome-level, independent of the active wrapper)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from labguard.compiler.compiler import Compiler
from labguard.compiler.monitor import CompiledMonitor
from labguard.envs.policies import PolicyConfig
from labguard.envs.runner import EpisodeResult, OfflineChecker, run_episode, seed_std, summarize
from labguard.envs.tasks import LEVELS, get_task
from labguard.envs.toy_lab import ToyLabEnv
from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import LabGuardIR
from labguard.runtime.rules import DEFAULT_RULES_PATH, gold_irs, load_rules
from labguard.runtime.shield import SafetyShield, ShieldConfig

RUNTIME_CONFIGS: dict[str, ShieldConfig | None] = {
    "Base": None,
    "Full": ShieldConfig(semantic=True, kinematic=True, planning=True, risk=True, logging=True),
}

ABLATION_CONFIGS: dict[str, ShieldConfig | None] = {
    "Base (no safety)": None,
    "Planning only": ShieldConfig(semantic=False, kinematic=False, planning=True, risk=False),
    "Semantic only": ShieldConfig(semantic=True, kinematic=False, planning=False, risk=False),
    "Action Filter only": ShieldConfig(semantic=False, kinematic=True, planning=False, risk=False),
    "Risk Scoring only": ShieldConfig(semantic=False, kinematic=False, planning=False, risk=True),
    "Full LabGuard": ShieldConfig(semantic=True, kinematic=True, planning=True, risk=True),
}

ABLATION_TASK = "PourStress"


def all_tasks(levels: Iterable[str] | None = None) -> list[str]:
    return [t for lv in (levels or LEVELS) for t in LEVELS[lv]]


def default_monitors(irs: Sequence[LabGuardIR] | None = None, registry: Registry | None = None
                     ) -> list[CompiledMonitor]:
    registry = registry or default_registry()
    irs = list(irs) if irs is not None else gold_irs(load_rules(DEFAULT_RULES_PATH))
    monitors, _ = Compiler(registry).compile_many(irs)
    return monitors


def episode_seeds(seeds: Sequence[int], episodes: int) -> list[int]:
    return [s * 1000 + k for s in seeds for k in range(episodes)]


def run_config(task: str, config: ShieldConfig | None, monitors: list[CompiledMonitor], seeds: Sequence[int],
               episodes: int, policy_config: PolicyConfig | None = None, checker: OfflineChecker | None = None,
               registry: Registry | None = None, config_name: str = "", record: bool = False
               ) -> list[EpisodeResult]:
    registry = registry or default_registry()
    shield = SafetyShield(monitors, config, registry) if config is not None else None
    env = ToyLabEnv(registry)
    return [run_episode(task, shield, seed=es, policy_config=policy_config, checker=checker, env=env,
                        config_name=config_name, record=record)
            for es in episode_seeds(seeds, episodes)]


def _summary(results: list[EpisodeResult]) -> dict[str, Any]:
    out = summarize(results)
    out["SR_std"] = seed_std(results, "success")
    return out


def run_runtime(tasks: Iterable[str] | None = None, seeds: Sequence[int] = (0, 1, 2), episodes: int = 10,
                irs: Sequence[LabGuardIR] | None = None,
                configs: Mapping[str, ShieldConfig | None] = RUNTIME_CONFIGS,
                policy_config: PolicyConfig | None = None, registry: Registry | None = None,
                keep_episodes: bool = False, log: Callable[[str], None] | None = print) -> dict[str, Any]:
    registry = registry or default_registry()
    monitors = default_monitors(irs, registry)
    tasks = list(tasks) if tasks is not None else all_tasks()
    per_task: dict[str, dict[str, Any]] = {}
    episodes_out: list[dict[str, Any]] = []
    for task in tasks:
        row: dict[str, Any] = {"level": get_task(task).level}
        for name, cfg in configs.items():
            t0 = time.perf_counter()
            res = run_config(task, cfg, monitors, seeds, episodes, policy_config, registry=registry,
                             config_name=name)
            row[name] = _summary(res)
            row[name]["wall_s"] = time.perf_counter() - t0
            if keep_episodes:
                episodes_out += [r.to_dict() for r in res]
        names = list(configs)
        if len(names) >= 2:
            row["dSR"] = row[names[-1]]["SR"] - row[names[0]]["SR"]
        per_task[task] = row
        if log:
            log(f"{row['level']} {task:16s} " + "  ".join(
                f"{n}: SR={row[n]['SR']:.3f}±{row[n]['SR_std']:.3f} IR={row[n]['IR'] * 100:.2f}% "
                f"VR={row[n]['VR_logged']:.2f} unsafe={row[n]['unsafe_event_rate']:.2f}" for n in names))
    by_level: dict[str, dict[str, float]] = {}
    for lv in sorted({r["level"] for r in per_task.values()}):
        rows = [r for r in per_task.values() if r["level"] == lv]
        by_level[lv] = {f"{n}_SR": sum(r[n]["SR"] for r in rows) / len(rows) for n in configs}
    out: dict[str, Any] = {"tasks": per_task, "levels": by_level, "seeds": list(seeds), "episodes": episodes,
                           "n_monitors": len(monitors)}
    if keep_episodes:
        out["episode_results"] = episodes_out
    return out


def run_ablation(task: str = ABLATION_TASK, seeds: Sequence[int] = (0, 1, 2), episodes: int = 10,
                 irs: Sequence[LabGuardIR] | None = None,
                 configs: Mapping[str, ShieldConfig | None] = ABLATION_CONFIGS,
                 policy_config: PolicyConfig | None = None, registry: Registry | None = None,
                 log: Callable[[str], None] | None = print) -> dict[str, Any]:
    registry = registry or default_registry()
    monitors = default_monitors(irs, registry)
    checker = OfflineChecker(default_monitors(None, registry))  # gold oracle, independent of `irs`
    rows: dict[str, Any] = {}
    for name, cfg in configs.items():
        res = run_config(task, cfg, monitors, seeds, episodes, policy_config, checker=checker,
                         registry=registry, config_name=name)
        rows[name] = _summary(res)
        rows[name]["violated_rules"] = _merge_counts(r.violated_rules for r in res)
        if log:
            s = rows[name]
            log(f"{name:20s} SR={s['SR']:.2f} IR={s['IR_with_plan'] * 100:.2f}% VR={s['VR_offline']:.2f} "
                f"Risk={s['Risk']:.3f} CR={s['CR']:.2f} EL={s['EL']:.0f} TCT={s['TCT']:.1f}s")
    return {"task": task, "configs": rows, "seeds": list(seeds), "episodes": episodes}


def _merge_counts(dicts: Iterable[Mapping[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
