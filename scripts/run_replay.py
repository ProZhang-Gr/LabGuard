"""Replay diagnostics: 41 held-out rules x scripted violating / compliant scenarios (80).

    python scripts/run_replay.py
    python scripts/run_replay.py --specs regex knn configs/grounder_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.bench.io import DEFAULT_BENCH_DIR  # noqa: E402
from labguard.eval.grounding import fit_grounder  # noqa: E402
from labguard.eval.replay import build_replay_set, run_replay  # noqa: E402
from labguard.eval.reporting import replay_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", nargs="*", default=["regex", "knn", "hybrid:knn"])
    ap.add_argument("--bench", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--split", default="random")
    ap.add_argument("--rules", type=int, default=41)
    ap.add_argument("--scenarios", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="results/replay")
    args = ap.parse_args()

    rules = build_replay_set(args.bench, args.split, args.rules, args.scenarios, args.seed)
    print(f"{len(rules)} rules, {sum(len(r.cases) for r in rules)} scenarios")
    results = {"No monitors": run_replay(None, rules)}
    for spec in args.specs:
        g = fit_grounder(spec, args.split, bench_dir=args.bench, root=args.root)
        results[spec] = run_replay(g, rules)
    results["Gold"] = run_replay("gold", rules)
    table = replay_table(results)
    print(f"\n## Replay diagnostics\n\n{table}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.md").write_text(table + "\n", encoding="utf-8")
    (out / "replay.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
