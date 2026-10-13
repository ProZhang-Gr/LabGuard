"""Runtime monitor execution on the L1-L4 task suite (Table 9, Fig. 3): Base vs Full LabGuard.

    python scripts/run_runtime.py                          # 3 seeds x 10 episodes per task
    python scripts/run_runtime.py --episodes 30 --levels L2 L3
    python scripts/run_runtime.py --grounder regex         # monitors from grounded instead of gold IRs

Runs in ToyLabEnv with the NoisyScriptedPolicy (stand-ins for LabUtopia / ACT); see
integrations/labutopia for running the same shield inside Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.envs.policies import PolicyConfig  # noqa: E402
from labguard.eval.reporting import runtime_table  # noqa: E402
from labguard.eval.runtime_eval import all_tasks, run_runtime  # noqa: E402
from labguard.runtime.rules import gold_irs, ground_rules, load_rules  # noqa: E402
from labguard.runtime.shield import ShieldConfig  # noqa: E402


def load_runtime_config(path: str | None) -> tuple[ShieldConfig, dict]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    return ShieldConfig.from_dict((cfg or {}).get("shield")), (cfg or {}).get("eval", {})


def rule_irs(rules_path: str | None, grounder: str | None):
    rules = load_rules(rules_path)
    if not grounder:
        return gold_irs(rules)
    from labguard.eval.grounding import fit_grounder

    return ground_rules(rules, fit_grounder(grounder))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "runtime.yaml"))
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--error-rate", type=float, default=None)
    ap.add_argument("--rules", default=None, help="runtime rule YAML (default: package rules)")
    ap.add_argument("--grounder", default=None, help="ground the rules with this spec instead of gold IR")
    ap.add_argument("--out", default="results/runtime")
    args = ap.parse_args()

    shield_cfg, ev = load_runtime_config(args.config)
    policy = PolicyConfig(**{**ev.get("policy", {}), **({"error_rate": args.error_rate}
                                                         if args.error_rate is not None else {})})
    tasks = args.tasks or all_tasks(args.levels)
    res = run_runtime(tasks, seeds=tuple(args.seeds or ev.get("seeds", (0, 1, 2))),
                      episodes=args.episodes or ev.get("episodes", 10), irs=rule_irs(args.rules, args.grounder),
                      configs={"Base": None, "Full": shield_cfg}, policy_config=policy)
    table = runtime_table(res)
    print(f"\n## Runtime (Table 9)\n\n{table}\n\nper level: {json.dumps(res['levels'])}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime.md").write_text(table + "\n", encoding="utf-8")
    (out / "runtime.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
