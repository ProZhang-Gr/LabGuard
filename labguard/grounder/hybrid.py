"""LabGuard-Hybrid: learned grounder + rule-based normalisation (Sec. 3.4, 5.2).

The learned component (LoRA, optionally with constrained decoding) predicts
the full IR; the rule-based layer then

  * snaps categorical fields (type, tags, severity, intervention) to the
    closed vocabularies, falling back to the regex extractor when a value is
    unusable,
  * resolves substances through the registry and falls back to regex
    extraction for the substance slot when the prediction contains
    unresolvable names or misses chemicals that the rule text names
    ("regex fallback for the substance slot"),
  * repairs predicates that do not parse or use unknown literals (alias ->
    id, nearest vocabulary literal, balanced parentheses) and, as a last
    resort, instantiates the regex template of the predicted type.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence

from labguard.grounder.base import Grounder, clean_ir
from labguard.grounder.constrained import IRGrammar
from labguard.grounder.regex_grounder import (
    build_predicate,
    classify_type,
    extract_intervention,
    extract_severity,
    extract_substances,
    extract_tags,
)
from labguard.ir.predicate import Predicate
from labguard.ir.registry import Registry
from labguard.ir.schema import LabGuardIR, sort_tags
from labguard.vocab import CONSTRAINT_TYPES, INTERVENTIONS, SEVERITIES, TASK_TAGS

if TYPE_CHECKING:
    from labguard.bench.example import BenchExample


def snap(value: str, vocab: Sequence[str], cutoff: float = 0.75) -> str | None:
    if value in vocab:
        return value
    m = difflib.get_close_matches(value.replace(" ", "_"), list(vocab), n=1, cutoff=cutoff)
    return m[0] if m else None


class HybridGrounder(Grounder):
    name = "hybrid"

    def __init__(self, neural: Grounder, registry: Registry | None = None, substance_fallback: bool = True,
                 repair_predicate: bool = True) -> None:
        super().__init__(registry or neural.registry)
        self.neural = neural
        self.substance_fallback = substance_fallback
        self.repair_predicate = repair_predicate
        self.trainable = neural.trainable
        self.grammar = IRGrammar(self.registry)
        self.stats: dict[str, int] = {"substance_fallback": 0, "predicate_repair": 0, "predicate_template": 0,
                                      "type_fallback": 0}

    def fit(self, train: Sequence["BenchExample"], dev: Sequence["BenchExample"] | None = None) -> "HybridGrounder":
        self.neural.fit(train, dev)
        return self

    def ground_batch(self, texts: Sequence[str]) -> list[LabGuardIR]:
        preds = self.neural.ground_batch(texts)
        self.last_raw = list(getattr(self.neural, "last_raw", []))
        return [self.normalize(t, p) for t, p in zip(texts, preds)]

    def ground(self, text: str) -> LabGuardIR:
        return self.ground_batch([text])[0]

    # -- normalisation -----------------------------------------------------------------
    def normalize(self, text: str, ir: LabGuardIR) -> LabGuardIR:
        ir = clean_ir(ir, self.registry)
        ctype = snap(ir.constraint_type, CONSTRAINT_TYPES)
        if ctype is None:
            ctype = classify_type(text)
            self.stats["type_fallback"] += 1
        tags = sort_tags(t for t in (snap(x, TASK_TAGS, 0.8) for x in ir.action_scope) if t)
        if not tags:
            tags = extract_tags(text, ctype)
        sev = snap(ir.severity, SEVERITIES) or extract_severity(text)
        inter = snap(ir.intervention, INTERVENTIONS) or extract_intervention(text, ctype, sev)
        subs = self._substances(text, ir, ctype)
        pred = self._predicate(text, ir.predicate, ctype, subs)
        return replace(ir, constraint_type=ctype, substances=subs, predicate=pred, action_scope=tags,
                       severity=sev, intervention=inter)

    def _substances(self, text: str, ir: LabGuardIR, ctype: str) -> tuple[str, ...]:
        resolved = [self.registry.resolve(s) for s in ir.substances]
        known = [s for s in resolved if s is not None]
        if not self.substance_fallback:
            return tuple(self.registry.sort_ids(known))
        mentioned = {s for s, _ in self.registry.find_mentions(text) if self.registry.is_chemical(s)}
        predicted = {s for s in known if self.registry.is_chemical(s)}
        if None in resolved or (mentioned and not predicted) or (mentioned - predicted and predicted <= mentioned):
            self.stats["substance_fallback"] += 1
            return extract_substances(text, self.registry, ctype, ir.predicate)
        return tuple(self.registry.sort_ids(known))

    def _predicate(self, text: str, pred: str, ctype: str, subs: tuple[str, ...]) -> str:
        if self._ok(pred) or not self.repair_predicate:
            return pred
        fixed = self._repair(pred)
        if fixed and self._ok(fixed):
            self.stats["predicate_repair"] += 1
            return fixed
        self.stats["predicate_template"] += 1
        return build_predicate(ctype, text, subs, self.registry)

    def _ok(self, pred: str) -> bool:
        p = Predicate.try_parse(pred, self.registry)
        return p is not None and not p.vocabulary_errors()

    def _repair(self, pred: str) -> str | None:
        if not pred.strip():
            return None
        s = pred.strip()
        s += ")" * max(0, s.count("(") - s.count(")"))
        s = re.sub(r"\"", "'", s)
        literals = sorted(self.grammar.literals)

        def lit(m: re.Match[str]) -> str:
            v = m.group(1)
            if v in self.grammar.literals:
                return m.group(0)
            sid = self.registry.resolve(v)
            if sid is not None:
                return f"'{sid}'"
            near = difflib.get_close_matches(v.lower().replace(" ", "_"), literals, n=1, cutoff=0.7)
            return f"'{near[0]}'" if near else m.group(0)

        return re.sub(r"'([^']*)'", lit, s)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "neural": self.neural.describe(), "substance_fallback": self.substance_fallback,
                "repair_predicate": self.repair_predicate}
