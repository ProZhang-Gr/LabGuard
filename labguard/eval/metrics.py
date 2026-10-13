"""Grounding metrics (Sec. 5.1, Table 2) and small aggregation helpers.

All metrics compare a predicted IR with the gold IR of the same rule after
`LabGuardIR.normalized()` (registry-resolved substances, sorted tags,
lower-cased categorical fields):

  type_acc       exact constraint-type match
  subst_f1       set F1 over substance / entity ids (both empty -> 1)
  tag_f1         set F1 over task tags T (both empty -> 1)
  pred_soft      token-level F1 between the predicate strings (multiset over
                 `predicate_tokens`; surface-level, so "temp > 80" and
                 "80 < temp" do not fully match, as discussed in Sec. 5.3)
  pred_em        canonical-form equality of the parsed predicates
  severity_acc / intervention_acc
  schema_valid   all fields in the closed vocabularies, substances resolvable,
                 predicate parses and uses only predicate-language vocabulary
  compile_ready  the IR passes every compiler stage including lowering

Per-example scores are averaged; `aggregate` reports mean and std over folds
or seeds (the "mean±std" of Table 2).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from labguard.compiler.compiler import Compiler
from labguard.ir.predicate import canonicalize, predicate_tokens
from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import LabGuardIR, validate_ir

GROUNDING_KEYS: tuple[str, ...] = (
    "type_acc", "subst_f1", "tag_f1", "pred_soft", "pred_em", "severity_acc", "intervention_acc",
    "schema_valid", "compile_ready",
)


# ----------------------------------------------------------------------------
# Primitive scores
# ----------------------------------------------------------------------------
def set_f1(pred: Iterable[str], gold: Iterable[str]) -> float:
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    return 2.0 * tp / (len(p) + len(g))


def token_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = sum((Counter(pred) & Counter(gold)).values())
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(gold)
    return 2 * prec * rec / (prec + rec)


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def predicate_em(pred: str, gold: str, registry: Registry | None = None) -> bool:
    cp = canonicalize(pred, registry)
    return cp is not None and cp == canonicalize(gold, registry)


# ----------------------------------------------------------------------------
# Grounding
# ----------------------------------------------------------------------------
def example_scores(pred: LabGuardIR, gold: LabGuardIR, registry: Registry | None = None,
                   compiler: Compiler | None = None) -> dict[str, float]:
    registry = registry or default_registry()
    compiler = compiler or Compiler(registry)
    p, g = pred.normalized(registry), gold.normalized(registry)
    _, err = compiler.try_compile(pred)
    return {
        "type_acc": float(p.constraint_type == g.constraint_type),
        "subst_f1": set_f1(p.substances, g.substances),
        "tag_f1": set_f1(p.action_scope, g.action_scope),
        "pred_soft": token_f1(predicate_tokens(pred.predicate), predicate_tokens(gold.predicate)),
        "pred_em": float(predicate_em(pred.predicate, gold.predicate, registry)),
        "severity_acc": float(p.severity == g.severity),
        "intervention_acc": float(p.intervention == g.intervention),
        "schema_valid": float(not validate_ir(pred, registry)),
        "compile_ready": float(err is None),
    }


def grounding_scores(preds: Sequence[LabGuardIR], golds: Sequence[LabGuardIR],
                     registry: Registry | None = None, per_example: bool = False) -> dict[str, Any]:
    if len(preds) != len(golds):
        raise ValueError(f"{len(preds)} predictions for {len(golds)} gold IRs")
    registry = registry or default_registry()
    compiler = Compiler(registry)
    rows = [example_scores(p, g, registry, compiler) for p, g in zip(preds, golds)]
    out: dict[str, Any] = {k: (sum(r[k] for r in rows) / len(rows) if rows else math.nan) for k in GROUNDING_KEYS}
    out["n"] = len(rows)
    if per_example:
        out["examples"] = rows
    return out


def confusion(preds: Sequence[LabGuardIR], golds: Sequence[LabGuardIR], field: str = "constraint_type"
              ) -> dict[str, dict[str, int]]:
    """gold value -> predicted value -> count (for error analysis)."""
    out: dict[str, dict[str, int]] = {}
    for p, g in zip(preds, golds):
        row = out.setdefault(getattr(g, field), {})
        row[getattr(p, field)] = row.get(getattr(p, field), 0) + 1
    return out


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def mean_std(values: Sequence[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan, math.nan
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, math.sqrt(var)


def aggregate(runs: Sequence[Mapping[str, Any]], keys: Iterable[str] | None = None) -> dict[str, dict[str, float]]:
    """{key: {"mean": .., "std": ..}} over a list of per-fold / per-seed result dicts."""
    keys = list(keys) if keys is not None else [k for k, v in runs[0].items() if isinstance(v, (int, float))] \
        if runs else []
    out = {}
    for k in keys:
        m, s = mean_std([float(r[k]) for r in runs if k in r and r[k] is not None])
        out[k] = {"mean": m, "std": s}
    return out


def fmt_pm(stat: Mapping[str, float], scale: float = 100.0, digits: int = 1) -> str:
    m, s = stat.get("mean", math.nan), stat.get("std", math.nan)
    if m is None or math.isnan(m):
        return "-"
    return f"{m * scale:.{digits}f}±{s * scale:.{digits}f}"
