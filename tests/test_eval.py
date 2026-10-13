"""Evaluation protocols: grounding metrics, stress benchmark (Table 3), replay diagnostics."""

import pytest

from labguard.eval.metrics import grounding_scores
from labguard.eval.replay import run_replay
from labguard.eval.stress import build_scenarios, run_stress_config, validate_scenarios


def test_grounding_scores_gold_is_perfect(examples):
    golds = [e.gold for e in examples[:120]]
    sc = grounding_scores(golds, golds)
    for k in ("type_acc", "subst_f1", "tag_f1", "pred_soft", "pred_em", "schema_valid", "compile_ready"):
        assert sc[k] == pytest.approx(1.0), k


def test_stress_benchmark_is_consistent():
    sc = build_scenarios()
    assert len(sc) == 100
    assert sum(s.unsafe for s in sc) == 70
    assert validate_scenarios(sc) == []


def test_stress_gold_matches_table3():
    c = run_stress_config("gold")["report"]["all"]
    assert (c["TP"], c["FN"], c["FP"]) == (54, 16, 4)


def test_stress_no_safety_catches_nothing():
    c = run_stress_config("none")["report"]["all"]
    assert c["TP"] == 0 and c["FN"] == 70


def test_replay_gold_is_perfect():
    res = run_replay("gold")
    assert (res["summary"]["n_rules"], res["summary"]["n_scenarios"]) == (41, 80)
    for k in ("check_acc", "decision_acc", "catch", "precision", "safety"):
        assert res["summary"][k] == pytest.approx(1.0), k


def test_replay_set_independent_of_hash_seed():
    """Scenario sampling must not depend on set iteration order (PYTHONHASHSEED)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = ("from labguard.eval.replay import build_replay_set\n"
            "rs = build_replay_set(n_rules=12, n_scenarios=24)\n"
            "print(repr([(r.seed_id, [(c.violating, c.action.describe(), sorted(c.state.ppe),"
            " sorted((e.id, sorted(e.flags.items())) for e in c.state.equipment.values())) for c in r.cases])"
            " for r in rs]))")
    root = Path(__file__).resolve().parents[1]
    outs = []
    for h in ("0", "123"):
        env = {**os.environ, "PYTHONHASHSEED": h, "PYTHONPATH": str(root)}
        outs.append(subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True,
                                   text=True, check=True).stdout)
    assert outs[0] == outs[1]
