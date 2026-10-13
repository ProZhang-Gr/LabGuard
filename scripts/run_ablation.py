"""Module ablation on the controlled PourStress scenario (Table 5).

    python scripts/run_ablation.py
    python scripts/run_ablation.py --episodes 30 --error-rate 0.3

VR is rescored offline with the gold compiled-monitor checker (+ kinematic checks), so the Base row
is comparable to rows with an active safety layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.envs.policies import PolicyConfig  # noqa: E402
from labguard.eval.reporting import ablation_table  # noqa: E402
from labguard.eval.runtime_eval import ABLATION_TASK, run_ablation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=ABLATION_TASK)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--error-rate", type=float, default=0.2)
    ap.add_argument("--out", default="results/ablation")
    args = ap.parse_args()

    res = run_ablation(args.task, tuple(args.seeds), args.episodes,
                       policy_config=PolicyConfig(error_rate=args.error_rate))
    table = ablation_table(res)
    print(f"\n## Module ablation (Table 5)\n\n{table}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.md").write_text(table + "\n", encoding="utf-8")
    (out / "ablation.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
