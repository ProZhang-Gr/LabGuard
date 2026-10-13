"""Grounding evaluation (Table 2): random / source-held-out / chemistry-held-out protocols.

    python scripts/eval_grounding.py --spec regex --protocol all
    python scripts/eval_grounding.py --spec knn hybrid:knn --protocol random source
    python scripts/eval_grounding.py --spec configs/grounder_lora.yaml --protocol source   # needs a GPU
    python scripts/eval_grounding.py --spec llm:Qwen/Qwen3-32B --protocol random           # OpenAI-compatible API

Trainable grounders are fitted per fold / seed (LoRA adapters are cached under --root and reused).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.bench.io import DEFAULT_BENCH_DIR  # noqa: E402
from labguard.eval.grounding import run_protocol  # noqa: E402
from labguard.eval.reporting import grounding_table  # noqa: E402

PROTOCOLS = ("random", "source", "chemistry")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", nargs="+", default=["regex", "knn", "hybrid:knn"], help="grounder specs / config files")
    ap.add_argument("--protocol", nargs="+", default=["random"], help=f"{PROTOCOLS} or 'all' or a split name")
    ap.add_argument("--bench", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="override PROTOCOL_SEEDS")
    ap.add_argument("--root", default=None, help="root directory for fold-specific adapters / checkpoints")
    ap.add_argument("--retrain", action="store_true", help="refit even if an adapter exists")
    ap.add_argument("--keep-predictions", action="store_true")
    ap.add_argument("--out", default="results/grounding")
    args = ap.parse_args()

    protocols = list(PROTOCOLS) if args.protocol == ["all"] else args.protocol
    out = Path(args.out)
    for protocol in protocols:
        results = {}
        for spec in args.spec:
            tag = Path(spec).stem if spec.endswith((".yaml", ".yml", ".json")) else spec.replace(":", "_")
            results[spec] = run_protocol(spec, protocol, args.bench, seeds=args.seeds, root=args.root,
                                         out_path=out / f"{protocol}__{tag}.json", reuse=not args.retrain,
                                         keep_predictions=args.keep_predictions)
        print(f"\n## Grounding, protocol = {protocol}\n")
        print(grounding_table(results))
        (out / f"{protocol}__table.md").write_text(grounding_table(results) + "\n", encoding="utf-8")
        print(json.dumps({s: r["summary"]["pred_em"] for s, r in results.items()}))


if __name__ == "__main__":
    main()
