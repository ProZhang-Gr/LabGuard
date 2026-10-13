"""Train fold-specific grounder checkpoints (LoRA Qwen3-8B or SciBERT) and evaluate them.

    python scripts/train_grounder.py --config configs/grounder_lora.yaml --protocol random
    python scripts/train_grounder.py --config configs/grounder_lora.yaml --protocol source chemistry \
        --root runs/lora/qwen3-8b
    python scripts/train_grounder.py --config configs/grounder_scibert.yaml --protocol random

Requires a CUDA build of torch for Qwen3-8B (bitsandbytes for `load_in_4bit`).  One adapter is
written per split and seed: <root>/<split>/seed<k>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.bench.io import DEFAULT_BENCH_DIR  # noqa: E402
from labguard.eval.grounding import run_protocol  # noqa: E402
from labguard.eval.reporting import grounding_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="grounder config (YAML/JSON) or spec string")
    ap.add_argument("--protocol", nargs="+", default=["random"])
    ap.add_argument("--bench", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--root", default=None, help="adapter root (default: the config's adapter / checkpoint)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--resume", action="store_true", help="skip folds whose adapter already exists")
    ap.add_argument("--out", default="results/grounding")
    args = ap.parse_args()

    tag = Path(args.config).stem
    for protocol in args.protocol:
        res = run_protocol(args.config, protocol, args.bench, seeds=args.seeds, root=args.root,
                           out_path=Path(args.out) / f"{protocol}__{tag}.json", reuse=args.resume,
                           dev_frac=args.dev_frac, keep_predictions=True)
        print(f"\n## {tag}, protocol = {protocol}\n")
        print(grounding_table({tag: res}))


if __name__ == "__main__":
    main()
