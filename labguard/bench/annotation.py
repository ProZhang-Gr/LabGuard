"""Simulated five-annotator review with majority adjudication (Appendix D.1).

The paper reports that five annotators (chemistry / robotics background)
reviewed every seed label and paraphrase, that final labels use simple
majority agreement (>= 3 of 5) and that items without a majority are
revised before inclusion.  We cannot reproduce human annotation, so this
module simulates it: every annotator reproduces the drafted label up to
field-level slips whose rate grows with item difficulty.  The simulation
exercises the adjudication protocol and yields agreement statistics
(per-field Fleiss' kappa, full-agreement and revision rates); it does not
change the gold labels of the released corpus.

With real annotations, call `adjudicate` on the collected label lists.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Hashable, Sequence

import numpy as np

from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import LabGuardIR, sort_tags
from labguard.vocab import SEVERITIES, TASK_TAGS

FIELDS: tuple[str, ...] = ("constraint_type", "substances", "predicate", "action_scope", "severity",
                           "intervention")

# Plausible confusions between neighbouring constraint types
CONFUSABLE_TYPES: dict[str, tuple[str, ...]] = {
    "chemical_incompatibility": ("storage_constraint", "waste_handling"),
    "storage_constraint": ("chemical_incompatibility",),
    "exposure_limit": ("ventilation_requirement", "ppe_requirement"),
    "ventilation_requirement": ("exposure_limit",),
    "ppe_requirement": ("exposure_limit",),
    "fire_safety": ("temperature_constraint", "workspace_zone"),
    "temperature_constraint": ("fire_safety", "equipment_check"),
    "spill_prevention": ("velocity_limit", "container_capacity"),
    "velocity_limit": ("spill_prevention",),
    "container_capacity": ("spill_prevention",),
    "equipment_check": ("sequence_constraint",),
    "sequence_constraint": ("equipment_check",),
    "waste_handling": ("chemical_incompatibility",),
    "workspace_zone": ("fire_safety",),
    "force_limit": ("velocity_limit",),
}
_NUM = re.compile(r"(?<![\w.'])(\d+(?:\.\d+)?)(?![\w'])")
_FLIP = {"<=": "<", "<": "<=", ">=": ">", ">": ">="}


@dataclass
class AnnotatorModel:
    """Per-field slip probabilities of one simulated annotator on an easy item."""

    slip: dict[str, float] = field(default_factory=lambda: {
        "constraint_type": 0.02, "substances": 0.015, "predicate": 0.03, "action_scope": 0.04,
        "severity": 0.06, "intervention": 0.02})
    difficulty_scale: dict[str, float] = field(default_factory=lambda: {"easy": 1.0, "medium": 1.4, "hard": 2.0})
    paraphrase_scale: float = 1.2


def _perturb_predicate(pred: str, rng: np.random.Generator) -> str:
    nums = list(_NUM.finditer(pred))
    ops = [op for op in _FLIP if op in pred]
    if nums and (not ops or rng.random() < 0.6):
        m = nums[int(rng.integers(len(nums)))]
        v = float(m.group(1))
        nv = v * float(rng.choice([0.5, 0.8, 1.25, 2.0]))
        nv_s = str(int(round(nv))) if v >= 5 else f"{nv:.2g}"
        return pred[:m.start(1)] + nv_s + pred[m.end(1):]
    if ops:
        op = max(ops, key=len)
        return pred.replace(op, _FLIP[op], 1)
    return pred if pred.startswith("not ") else f"not ({pred})"


def perturb_field(ir: LabGuardIR, name: str, rng: np.random.Generator, registry: Registry) -> LabGuardIR:
    if name == "constraint_type":
        return replace(ir, constraint_type=str(rng.choice(CONFUSABLE_TYPES[ir.constraint_type])))
    if name == "severity":
        k = SEVERITIES.index(ir.severity)
        nk = k + (1 if k == 0 else -1 if k == 2 else int(rng.choice([-1, 1])))
        return replace(ir, severity=SEVERITIES[nk])
    if name == "intervention":
        return replace(ir, intervention=str(rng.choice([i for i in ("log", "modify", "block") if i != ir.intervention])))
    if name == "action_scope":
        tags = list(ir.action_scope)
        if len(tags) > 1 and rng.random() < 0.6:
            tags.pop(int(rng.integers(len(tags))))
        else:
            tags.append(str(rng.choice([t for t in TASK_TAGS if t not in tags])))
        return replace(ir, action_scope=sort_tags(tags))
    if name == "substances":
        subs = list(ir.substances)
        if subs and rng.random() < 0.5:
            subs.pop(int(rng.integers(len(subs))))
        else:
            subs.append(str(rng.choice([s for s in registry.ids if s not in subs])))
        return replace(ir, substances=tuple(registry.sort_ids(subs)))
    if name == "predicate":
        return replace(ir, predicate=_perturb_predicate(ir.predicate, rng))
    raise KeyError(name)


def simulate_annotators(ir: LabGuardIR, difficulty: str, is_paraphrase: bool, rng: np.random.Generator,
                        n: int = 5, model: AnnotatorModel | None = None,
                        registry: Registry | None = None) -> list[LabGuardIR]:
    model = model or AnnotatorModel()
    registry = registry or default_registry()
    scale = model.difficulty_scale.get(difficulty, 1.0) * (model.paraphrase_scale if is_paraphrase else 1.0)
    labels = []
    for _ in range(n):
        lab = ir
        for name in FIELDS:
            if rng.random() < min(0.9, model.slip[name] * scale):
                lab = perturb_field(lab, name, rng, registry)
        labels.append(lab)
    return labels


def field_value(ir: LabGuardIR, name: str) -> Hashable:
    v = getattr(ir, name)
    return tuple(v) if isinstance(v, (list, tuple)) else v


@dataclass
class Adjudication:
    label: LabGuardIR
    votes: dict[str, list[Hashable]]
    majority: dict[str, bool]
    agreement: float  # fraction of annotator x field votes equal to the final label
    full_agreement: bool
    revised: bool  # some field had no >= 3/5 majority (or the majority contradicted the reviewed draft)


def adjudicate(labels: Sequence[LabGuardIR], draft: LabGuardIR | None = None, min_votes: int = 3) -> Adjudication:
    """Field-wise majority vote.  Fields without a majority fall back to the revised draft."""
    votes = {f: [field_value(lab, f) for lab in labels] for f in FIELDS}
    final: dict[str, Any] = {}
    majority: dict[str, bool] = {}
    revised = False
    for f, vs in votes.items():
        value, count = Counter(vs).most_common(1)[0]
        majority[f] = count >= min_votes
        if majority[f] and (draft is None or value == field_value(draft, f)):
            final[f] = value
        else:
            if draft is None:
                raise ValueError(f"no majority for field {f} and no draft to revise from")
            final[f] = field_value(draft, f)
            revised = True
    base = draft or labels[0]
    label = replace(base, **{f: final[f] for f in FIELDS})
    agree = sum(v == final[f] for f, vs in votes.items() for v in vs) / max(1, len(labels) * len(FIELDS))
    full = all(len(set(vs)) == 1 for vs in votes.values())
    return Adjudication(label, votes, majority, float(agree), full, revised)


def fleiss_kappa(items: Sequence[Sequence[Hashable]]) -> float:
    """Fleiss' kappa for N items rated by the same number of raters into nominal categories."""
    items = [list(x) for x in items if x]
    if not items:
        return float("nan")
    n = len(items[0])
    cats = sorted({c for x in items for c in x}, key=repr)
    idx = {c: k for k, c in enumerate(cats)}
    counts = np.zeros((len(items), len(cats)))
    for i, x in enumerate(items):
        for c in x:
            counts[i, idx[c]] += 1
    p_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    p_bar = float(np.mean(p_i))
    p_j = counts.sum(axis=0) / (len(items) * n)
    p_e = float(np.sum(p_j ** 2))
    return 1.0 if p_e >= 1.0 else (p_bar - p_e) / (1.0 - p_e)


def agreement_report(results: Sequence[Adjudication]) -> dict[str, Any]:
    if not results:
        return {}
    return {
        "items": len(results),
        "annotators": len(next(iter(results[0].votes.values()))),
        "full_agreement_rate": float(np.mean([r.full_agreement for r in results])),
        "revised_rate": float(np.mean([r.revised for r in results])),
        "mean_vote_agreement": float(np.mean([r.agreement for r in results])),
        "fleiss_kappa": {f: round(fleiss_kappa([r.votes[f] for r in results]), 4) for f in FIELDS},
        "majority_rate": {f: float(np.mean([r.majority[f] for r in results])) for f in FIELDS},
    }
