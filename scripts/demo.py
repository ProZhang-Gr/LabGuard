"""End-to-end LabGuard walk-through (Fig. 1): rule text -> LabGuard-IR -> monitor -> filtered action.

    python scripts/demo.py
    python scripts/demo.py --grounder hybrid:knn   # retrieval grounder fitted on the random split
    python scripts/demo.py --grounder gold         # use the gold IR of the rules
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labguard.core.action import Action  # noqa: E402
from labguard.envs.runner import run_episode  # noqa: E402
from labguard.envs.scene import default_scene  # noqa: E402
from labguard.eval.grounding import fit_grounder  # noqa: E402
from labguard.eval.metrics import example_scores  # noqa: E402
from labguard.ir.serialize import ir_to_text  # noqa: E402
from labguard.runtime.rules import load_rules  # noqa: E402
from labguard.runtime.shield import SafetyShield, ShieldConfig  # noqa: E402

DEMO_RULES = ("R01", "R06", "R07", "R09")  # incompatibility, pour rate, temperature, carrying speed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grounder", default="regex")
    ap.add_argument("--task", default="PourLiquid")
    args = ap.parse_args()

    rules = {r.rule_id: r for r in load_rules()}
    grounder = None if args.grounder == "gold" else fit_grounder(args.grounder)
    print("=" * 72, f"\n1) Grounding  (rule text -> LabGuard-IR, grounder={args.grounder})\n")
    irs = []
    for rid in DEMO_RULES:
        r = rules[rid]
        ir = r.gold if grounder is None else grounder.ground(r.text).with_meta(rule_id=rid, rule_text=r.text)
        irs.append(ir)
        sc = example_scores(ir, r.gold)
        wrong = [k for k in ("type_acc", "subst_f1", "tag_f1", "pred_em") if sc[k] < 1]
        verdict = "exact" if not wrong else "differs in " + ", ".join(wrong)
        print(f"[{rid}] {r.text}\n{ir_to_text(ir)}\n  vs gold: {verdict}\n")

    shield = SafetyShield.from_irs(irs, ShieldConfig(planning=False))
    print("=" * 72, "\n2) Compilation  (IR -> typed monitors)\n")
    print(shield.describe(), "\n")

    print("=" * 72, "\n3) Runtime filtering  (pi_safe(s_t, a_t))\n")
    s = default_scene(np.random.default_rng(0))
    s.ppe = {"gloves", "goggles", "lab_coat"}
    s.containers["flask_ethanol"].lid_open = True
    proposals = [
        Action("pour", "bottle_hcl", "beaker_bleach", volume=20.0, pour_rate=10.0),
        Action("pour", "beaker_water", "beaker_target", volume=40.0, pour_rate=60.0),
        Action("heat", "flask_ethanol", "hot_plate_1", target_temperature=90.0),
        Action("transport", "beaker_water", "general_shelf", speed=0.5),
        Action("pour", "beaker_water", "beaker_target", volume=40.0, pour_rate=15.0),
    ]
    for a in proposals:
        d = shield.filter(s, a)
        out = d.action.describe() if d.action is not None else "∅ (blocked)"
        print(f"{d.verdict:7s} {a.describe()}\n        -> {out}   fired={d.fired_rules}")

    print("\n" + "=" * 72, f"\n4) Closed loop  ({args.task}, weak scripted policy, 5 episodes)\n")
    full = SafetyShield.from_rules(load_rules(), config=ShieldConfig(planning=True))
    for name, sh in (("Base", None), ("LabGuard", full)):
        rs = [run_episode(args.task, sh, seed=k) for k in range(5)]
        print(f"{name:9s} success={sum(r.success for r in rs)}/5 hazards="
              f"{sorted({h for r in rs for h in r.hazards})} interventions={sum(r.interventions for r in rs)}")


if __name__ == "__main__":
    main()
