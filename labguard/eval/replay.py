"""Replay diagnostics for compiled monitors (Sec. 5.3, Table 6 "Replay diagnostics").

41 held-out rules (unique seed rules of the random test split, stratified by
constraint type) are paired with scripted scenarios sampled by
`ScenarioSampler`: for every rule one violating and one compliant
(s_t, a_t), labelled by the monitor compiled from the *gold* IR, truncated to
80 scenarios.  A grounder grounds the rule text; its IR is compiled and
replayed on the same scenarios.

Per rule (static, IR/monitor level)
  check_acc       the predicted IR compiles to the same check function
  scope_f1        set F1 between predicted and gold task scopes T
  param_recovery  share of the gold monitor parameters theta (comparison
                  limits within 1 %, call arguments) present in the prediction
Per scenario (dynamic, decision level)
  decision_acc    the predicted monitor flags exactly the violating scenarios
  catch           violating scenarios flagged / violating scenarios
  precision       flagged scenarios that violate / flagged scenarios
  safety          violating scenarios whose executed action is safe w.r.t. the
                  gold monitor (blocked, or repaired into compliance, or
                  logged when the gold rule only asks to log)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from labguard.bench.io import DEFAULT_BENCH_DIR, load_all_splits, load_examples
from labguard.compiler.compiler import Compiler
from labguard.compiler.monitor import CompiledMonitor
from labguard.eval.metrics import prf, set_f1
from labguard.eval.scenarios import Scenario, ScenarioSampler
from labguard.grounder.base import Grounder
from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import LabGuardIR

REPLAY_KEYS: tuple[str, ...] = ("check_acc", "scope_f1", "param_recovery", "decision_acc", "catch", "precision",
                                "f1", "safety")


@dataclass
class ReplayRule:
    seed_id: str
    example_id: str
    text: str
    gold: LabGuardIR
    cases: list[Scenario] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Replay set
# ----------------------------------------------------------------------------
def build_replay_set(bench_dir: str | Path = DEFAULT_BENCH_DIR, split: str = "random", n_rules: int = 41,
                     n_scenarios: int = 80, seed: int = 42, registry: Registry | None = None,
                     sampler: ScenarioSampler | None = None) -> list[ReplayRule]:
    registry = registry or default_registry()
    sampler = sampler or ScenarioSampler(registry)
    compiler = Compiler(registry)
    examples = load_examples(bench_dir)
    _, test = load_all_splits(bench_dir)[split].select(examples)
    first: dict[str, Any] = {}
    for e in test:
        first.setdefault(e.seed_id, e)
    by_type: dict[str, list[Any]] = defaultdict(list)
    seeds = sorted(first)
    random.Random(seed).shuffle(seeds)
    for sid in seeds:
        by_type[first[sid].gold.constraint_type].append(first[sid])
    # round-robin over constraint types -> stratified rule order
    order: list[Any] = []
    queues = [by_type[t] for t in sorted(by_type)]
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))

    rng = np.random.default_rng(seed)
    rules: list[ReplayRule] = []
    for e in order:
        if len(rules) >= n_rules:
            break
        mon, err = compiler.try_compile(e.gold)
        if mon is None:
            continue
        found = sampler.find_cases(mon, rng)
        if len(found) < 2:
            continue
        rules.append(ReplayRule(e.seed_id, e.id, e.text, e.gold, [found[True], found[False]]))
    # truncate to n_scenarios by dropping compliant cases from the last rules
    excess = sum(len(r.cases) for r in rules) - n_scenarios
    for r in reversed(rules):
        if excess <= 0:
            break
        r.cases = [c for c in r.cases if c.violating]
        excess -= 1
    return rules


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def _num(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


def param_recovery(pred: CompiledMonitor | None, gold: CompiledMonitor) -> float | None:
    g = gold.theta
    items = len(g["limits"]) + len(g["calls"])
    if items == 0:
        return None
    if pred is None:
        return 0.0
    p = pred.theta
    hit = 0
    for path, op, val in g["limits"]:
        gv = _num(val)
        for pp, po, pv in p["limits"]:
            if pp != path or po != op:
                continue
            pnum = _num(pv)
            if (gv is None and pv == val) or (gv is not None and pnum is not None
                                              and abs(pnum - gv) <= 0.01 * max(abs(gv), 1e-9)):
                hit += 1
                break
    hit += len(set(g["calls"]) & set(p["calls"]))
    return hit / items


def _safe_outcome(pred: CompiledMonitor | None, gold: CompiledMonitor, case: Scenario) -> tuple[bool, bool]:
    """(fired, outcome is safe w.r.t. the gold monitor) for one scenario."""
    if pred is None:
        return False, False
    res = pred(case.state, case.action)
    fired = res.violated is True
    if not case.violating:
        return fired, True
    if res.verdict == "block":
        return fired, True
    if res.verdict == "modify":
        return fired, gold.violated(case.state, res.action) is not True
    return fired, fired and gold.intervention == "log"


def replay_rule(rule: ReplayRule, pred_ir: LabGuardIR | None, compiler: Compiler,
                registry: Registry) -> dict[str, Any]:
    gold, _ = compiler.try_compile(rule.gold)
    assert gold is not None, rule.seed_id
    pred, err = compiler.try_compile(pred_ir) if pred_ir is not None else (None, "no prediction")
    gs = rule.gold.normalized(registry)
    ps = pred_ir.normalized(registry) if pred_ir is not None else None
    row: dict[str, Any] = {
        "seed_id": rule.seed_id,
        "type": rule.gold.constraint_type,
        "compiled": pred is not None,
        "error": None if err is None else str(err),
        "check_acc": float(pred is not None and pred.check.name == gold.check.name),
        "scope_f1": set_f1(ps.action_scope, gs.action_scope) if ps is not None else 0.0,
        "param_recovery": param_recovery(pred, gold),
        "cases": [],
    }
    for c in rule.cases:
        fired, safe = _safe_outcome(pred, gold, c)
        row["cases"].append({"violating": c.violating, "fired": fired, "safe": safe})
    return row


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cases = [c for r in rows for c in r["cases"]]
    tp = sum(c["violating"] and c["fired"] for c in cases)
    fn = sum(c["violating"] and not c["fired"] for c in cases)
    fp = sum(not c["violating"] and c["fired"] for c in cases)
    unsafe = [c for c in cases if c["violating"]]
    pr = [r["param_recovery"] for r in rows if r["param_recovery"] is not None]
    p = prf(tp, fp, fn)
    return {
        "n_rules": len(rows),
        "n_scenarios": len(cases),
        "n_violating": len(unsafe),
        "check_acc": sum(r["check_acc"] for r in rows) / len(rows) if rows else math.nan,
        "scope_f1": sum(r["scope_f1"] for r in rows) / len(rows) if rows else math.nan,
        "param_recovery": sum(pr) / len(pr) if pr else math.nan,
        "decision_acc": sum(c["fired"] == c["violating"] for c in cases) / len(cases) if cases else math.nan,
        "catch": p["recall"],
        "precision": p["precision"],
        "f1": p["f1"],
        "safety": sum(c["safe"] for c in unsafe) / len(unsafe) if unsafe else math.nan,
        "TP": tp, "FN": fn, "FP": fp,
    }


def replay_diagnostics(rules: Sequence[ReplayRule], preds: Sequence[LabGuardIR | None],
                       registry: Registry | None = None) -> dict[str, Any]:
    registry = registry or default_registry()
    compiler = Compiler(registry)
    rows = [replay_rule(r, p, compiler, registry) for r, p in zip(rules, preds)]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {"summary": summarize(rows), "by_type": {t: summarize(rs) for t, rs in sorted(by_type.items())},
            "rules": rows}


def run_replay(grounder: Grounder | str | None, rules: Sequence[ReplayRule] | None = None,
               registry: Registry | None = None, **build_kw: Any) -> dict[str, Any]:
    """`grounder` = "gold" (oracle IRs), None (no monitors) or a fitted Grounder."""
    registry = registry or default_registry()
    rules = list(rules) if rules is not None else build_replay_set(registry=registry, **build_kw)
    if grounder == "gold":
        preds: list[LabGuardIR | None] = [r.gold for r in rules]
    elif grounder is None or grounder == "none":
        preds = [None] * len(rules)
    else:
        preds = list(grounder.ground_batch([r.text for r in rules]))
    return replay_diagnostics(rules, preds, registry)
