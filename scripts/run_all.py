"""Run every CPU-feasible experiment of the paper and write results/REPORT.md.

    python scripts/run_all.py                    # full CPU run (~ a few minutes)
    python scripts/run_all.py --quick            # 1 seed / 3 episodes, random split only (smoke run)
    python scripts/run_all.py --specs regex knn hybrid:knn configs/grounder_lora.yaml   # + LoRA (GPU)

Sections: Table 2 (grounding), Table 3 (stress), Table 4 (propagation), replay diagnostics,
Table 9 / Fig. 3 (runtime L1-L4) and Table 5 (module ablation).  Raw JSON goes next to the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.bench.io import DEFAULT_BENCH_DIR  # noqa: E402
from labguard.envs.policies import PolicyConfig  # noqa: E402
from labguard.eval.grounding import fit_grounder, make_builder, protocol_splits, run_protocol  # noqa: E402
from labguard.eval.replay import build_replay_set, run_replay  # noqa: E402
from labguard.eval.reporting import (ablation_table, grounding_table, propagation_table, replay_table,  # noqa: E402
                                     runtime_table, stress_table)
from labguard.eval.runtime_eval import run_ablation, run_runtime  # noqa: E402
from labguard.eval.stress import run_stress_config, stress_propagation, validate_scenarios  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _name(spec: str) -> str:
    return Path(spec).stem if spec.endswith((".yaml", ".yml", ".json")) else spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", nargs="+", default=["regex", "knn", "hybrid:knn"])
    ap.add_argument("--bench", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--root", default=None, help="root directory for fold-specific adapters")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="runtime / ablation seeds")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--error-rate", type=float, default=0.2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["grounding", "stress", "propagation", "replay", "runtime", "ablation"])
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.episodes = [0], 3
    protocols = ["random"] if args.quick else ["random", "source", "chemistry"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw: dict[str, object] = {}
    md = ["# LabGuard reproduction report", "",
          f"Grounders: {', '.join(args.specs)}; runtime seeds {args.seeds} x {args.episodes} episodes, "
          f"policy error rate {args.error_rate}.", ""]
    timings: dict[str, float] = {}

    def section(key: str, title: str, body: str, t0: float) -> None:
        timings[key] = time.perf_counter() - t0
        md.extend([f"## {title}", "", body, ""])
        print(f"\n## {title}\n\n{body}\n", flush=True)

    if "grounding" not in args.skip:
        t0 = time.perf_counter()
        parts = []
        for protocol in protocols:
            res = {_name(s): run_protocol(s, protocol, args.bench, root=args.root, log=None) for s in args.specs}
            raw[f"grounding_{protocol}"] = res
            parts.append(f"### {protocol}\n\n{grounding_table(res)}")
        section("grounding", "Grounding (Table 2)", "\n\n".join(parts), t0)

    if "stress" not in args.skip:
        t0 = time.perf_counter()
        problems = validate_scenarios()
        res = {"No Safety": run_stress_config("none")}
        for spec in args.specs:
            res[_name(spec)] = run_stress_config(_name(spec), fit_grounder(spec, "random", bench_dir=args.bench,
                                                                           root=args.root))
        res["Gold"] = run_stress_config("gold")
        raw["stress"] = res
        body = stress_table(res) + ("\n\nScenario validation: OK" if not problems
                                    else "\n\nScenario validation problems:\n" + "\n".join(problems))
        section("stress", "Stress benchmark (Table 3)", body, t0)

    if "propagation" not in args.skip:
        t0 = time.perf_counter()
        trainable = [s for s in args.specs if s not in ("regex", "oracle")]
        res = {_name(s): stress_propagation(make_builder(s, root=args.root), protocol_splits(p, args.bench),
                                            args.bench, log=None)
               for s in trainable for p in (["random"] if args.quick else ["source"])}
        raw["propagation"] = res
        section("propagation", "Grounding -> stress propagation (Table 4)",
                propagation_table(res) if res else "(no trainable grounder)", t0)

    if "replay" not in args.skip:
        t0 = time.perf_counter()
        rules = build_replay_set(args.bench)
        res = {"No monitor": run_replay(None, rules), "Gold": run_replay("gold", rules)}
        for spec in args.specs:
            res[_name(spec)] = run_replay(fit_grounder(spec, "random", bench_dir=args.bench, root=args.root), rules)
        raw["replay"] = res
        n = res["Gold"]["summary"]
        section("replay", "Replay diagnostics",
                f"{n['n_rules']} held-out rules, {n['n_scenarios']} scripted scenarios "
                f"({n['n_violating']} violating).\n\n{replay_table(res)}", t0)

    policy = PolicyConfig(error_rate=args.error_rate)
    if "runtime" not in args.skip:
        t0 = time.perf_counter()
        res = run_runtime(seeds=args.seeds, episodes=args.episodes, policy_config=policy, log=None)
        raw["runtime"] = res
        lv = "\n".join(f"- {k}: Base SR {100 * v['Base_SR']:.1f}%, Full SR {100 * v['Full_SR']:.1f}%"
                       for k, v in res["levels"].items())
        section("runtime", "Runtime monitor execution (Table 9 / Fig. 3)", f"{runtime_table(res)}\n\n{lv}", t0)

    if "ablation" not in args.skip:
        t0 = time.perf_counter()
        res = run_ablation(seeds=args.seeds, episodes=args.episodes, policy_config=policy, log=None)
        raw["ablation"] = res
        section("ablation", f"Module ablation on {res['task']} (Table 5)", ablation_table(res), t0)

    md += ["## Wall-clock", "", "\n".join(f"- {k}: {v:.1f}s" for k, v in timings.items()), ""]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(raw, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
