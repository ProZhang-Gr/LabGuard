"""Builds LabGuard-Bench (203 seed rules -> 812 labeled examples) and its splits.

    python scripts/build_bench.py                      # template paraphraser, offline
    python scripts/build_bench.py --paraphraser llm:gpt-4o-mini --out data/labguard_bench_llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.bench import DEFAULT_BENCH_DIR, BenchConfig, build_bench, check_against_paper, save_bench  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--seed", type=int, default=0, help="generation seed")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--paraphraser", default="template", help='"template" or "llm[:model]"')
    ap.add_argument("--held-out-chemical", default="HCl")
    ap.add_argument("--no-annotate", action="store_true", help="skip the simulated five-annotator review")
    args = ap.parse_args()

    cfg = BenchConfig(seed=args.seed, paraphraser=args.paraphraser, held_out_chemical=args.held_out_chemical,
                      annotate=not args.no_annotate)
    bench = build_bench(cfg)
    paths = save_bench(bench, args.out, split_seed=args.split_seed)
    stats = bench.stats()
    issues = check_against_paper(stats)
    print(f"seeds={len(bench.seeds)} examples={len(bench.examples)}")
    print("types:", json.dumps(stats["type_counts"]))
    print("sources:", json.dumps(stats["source_seed_counts"]), "difficulty:", json.dumps(stats["difficulty_seed_counts"]))
    if bench.annotation:
        print("annotation:", json.dumps({k: bench.annotation[k] for k in
                                         ("full_agreement_rate", "revised_rate", "fleiss_kappa")}))
    print("paper check:", "OK" if not issues else issues)
    for name, p in paths.items():
        print(f"  {name:<11} {p}")


if __name__ == "__main__":
    main()
