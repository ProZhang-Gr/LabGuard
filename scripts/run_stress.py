"""Stress benchmark (Table 3) and source-held-out propagation (Table 4).

    python scripts/run_stress.py                               # No Safety / Regex / KNN / Hybrid(KNN) / Gold
    python scripts/run_stress.py --specs regex configs/grounder_lora.yaml
    python scripts/run_stress.py --propagation --specs knn hybrid:knn
    python scripts/run_stress.py --validate                    # sanity-check the 100 scenarios

Grounders for Table 3 are fitted on the random split; Table 4 re-grounds the same rule pool with
the four source-held-out models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.bench.io import DEFAULT_BENCH_DIR  # noqa: E402
from labguard.eval.grounding import fit_grounder, make_builder, protocol_splits  # noqa: E402
from labguard.eval.reporting import propagation_table, stress_category_table, stress_table  # noqa: E402
from labguard.eval.stress import run_stress_config, stress_propagation, validate_scenarios  # noqa: E402


def _name(spec: str) -> str:
    return Path(spec).stem if spec.endswith((".yaml", ".yml", ".json")) else spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", nargs="*", default=["regex", "knn", "hybrid:knn"])
    ap.add_argument("--bench", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--root", default=None)
    ap.add_argument("--propagation", action="store_true", help="Table 4 instead of Table 3")
    ap.add_argument("--protocol", default="source", help="splits used for --propagation")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default="results/stress")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.validate:
        problems = validate_scenarios()
        print("\n".join(problems) if problems else "all 100 stress scenarios behave as specified")
        sys.exit(1 if problems else 0)

    if args.propagation:
        results = {}
        for spec in args.specs:
            results[_name(spec)] = stress_propagation(make_builder(spec, root=args.root),
                                                      protocol_splits(args.protocol, args.bench), args.bench)
        table = propagation_table(results)
        print(f"\n## Stress propagation ({args.protocol}-held-out grounders)\n\n{table}")
        (out / f"propagation_{args.protocol}.md").write_text(table + "\n", encoding="utf-8")
        (out / f"propagation_{args.protocol}.json").write_text(json.dumps(results, indent=1, default=str),
                                                               encoding="utf-8")
        return

    results = {"No Safety": run_stress_config("none")}
    for spec in args.specs:
        g = fit_grounder(spec, "random", bench_dir=args.bench, root=args.root)
        results[_name(spec)] = run_stress_config(_name(spec), g)
    results["Gold"] = run_stress_config("gold")
    md = [f"## Stress benchmark (Table 3)\n\n{stress_table(results)}",
          f"## In-schema subset\n\n{stress_table(results, 'in_schema', paper=False)}",
          f"## Out-of-schema subset\n\n{stress_table(results, 'out_of_schema', paper=False)}",
          f"## Gold by hazard category\n\n{stress_category_table(results['Gold'])}"]
    print("\n\n".join(md))
    (out / "stress.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    (out / "stress.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
