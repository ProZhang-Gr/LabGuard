"""LabGuard-IR: z = (t, S, p, T, v, i)  (Sec. 3.2).

  t  constraint type      one of the 15 CONSTRAINT_TYPES
  S  substances/entities  set of canonical registry ids
  p  predicate            requirement over (state, action); see ir.predicate
  T  action scope         set of task tags the rule applies to (empty = all)
  v  severity             low | medium | high
  i  intervention mode    log | modify | block
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from labguard.ir.predicate import Predicate, PredicateError
from labguard.ir.registry import Registry, default_registry
from labguard.vocab import CONSTRAINT_TYPES, INTERVENTIONS, SEVERITIES, TASK_TAGS

IR_FIELDS: tuple[str, ...] = (
    "constraint_type", "substances", "predicate", "action_scope", "severity", "intervention",
)
# Short keys used by the line serialization (see ir.serialize)
SHORT_KEYS: dict[str, str] = {
    "constraint_type": "type",
    "substances": "substances",
    "predicate": "predicate",
    "action_scope": "scope",
    "severity": "severity",
    "intervention": "intervention",
}
_TAG_ORDER = {t: k for k, t in enumerate(TASK_TAGS)}


def sort_tags(tags: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(tags), key=lambda t: (_TAG_ORDER.get(t, len(_TAG_ORDER)), t)))


@dataclass(frozen=True)
class LabGuardIR:
    constraint_type: str
    substances: tuple[str, ...]
    predicate: str
    action_scope: tuple[str, ...]
    severity: str
    intervention: str
    rule_id: str | None = field(default=None, compare=False)
    rule_text: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "substances", tuple(self.substances))
        object.__setattr__(self, "action_scope", tuple(self.action_scope))

    # ------------------------------------------------------------------ dicts
    def to_dict(self, with_meta: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "constraint_type": self.constraint_type,
            "substances": list(self.substances),
            "predicate": self.predicate,
            "action_scope": list(self.action_scope),
            "severity": self.severity,
            "intervention": self.intervention,
        }
        if with_meta:
            d["rule_id"] = self.rule_id
            d["rule_text"] = self.rule_text
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabGuardIR":
        def listify(v: Any) -> tuple[str, ...]:
            if v is None:
                return ()
            if isinstance(v, str):
                return tuple(x.strip() for x in v.replace(",", ";").split(";") if x.strip())
            return tuple(str(x) for x in v)

        return cls(
            constraint_type=str(d.get("constraint_type", d.get("type", ""))).strip(),
            substances=listify(d.get("substances", d.get("S"))),
            predicate=str(d.get("predicate", d.get("p", ""))).strip(),
            action_scope=listify(d.get("action_scope", d.get("scope", d.get("T")))),
            severity=str(d.get("severity", "")).strip(),
            intervention=str(d.get("intervention", "")).strip(),
            rule_id=d.get("rule_id"),
            rule_text=d.get("rule_text"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def with_meta(self, rule_id: str | None = None, rule_text: str | None = None) -> "LabGuardIR":
        return replace(self, rule_id=rule_id if rule_id is not None else self.rule_id,
                       rule_text=rule_text if rule_text is not None else self.rule_text)

    # ------------------------------------------------------------- normalise
    def normalized(self, registry: Registry | None = None) -> "LabGuardIR":
        """Canonical form: resolved/sorted S, sorted T, canonical predicate.

        Unresolvable substances and unparsable predicates are kept verbatim so
        that comparison against gold still penalises them.
        """
        registry = registry or default_registry()
        subs = []
        for s in self.substances:
            sid = registry.resolve(s)
            subs.append(sid if sid is not None else s)
        known = [s for s in subs if s in registry.substances]
        unknown = sorted({s for s in subs if s not in registry.substances})
        pred = Predicate.try_parse(self.predicate, registry)
        return replace(
            self,
            constraint_type=self.constraint_type.strip().lower(),
            substances=tuple(registry.sort_ids(known)) + tuple(unknown),
            predicate=pred.canonical if pred else self.predicate.strip(),
            action_scope=sort_tags(t.strip().lower() for t in self.action_scope),
            severity=self.severity.strip().lower(),
            intervention=self.intervention.strip().lower(),
        )


def validate_ir(ir: LabGuardIR, registry: Registry | None = None) -> list[tuple[str, str]]:
    """Schema validation (compiler stage 1-3).  Returns a list of (stage, message)."""
    registry = registry or default_registry()
    errors: list[tuple[str, str]] = []
    if ir.constraint_type not in CONSTRAINT_TYPES:
        errors.append(("schema", f"unknown constraint type {ir.constraint_type!r}"))
    if ir.severity not in SEVERITIES:
        errors.append(("schema", f"unknown severity {ir.severity!r}"))
    if ir.intervention not in INTERVENTIONS:
        errors.append(("schema", f"unknown intervention {ir.intervention!r}"))
    for tag in ir.action_scope:
        if tag not in TASK_TAGS:
            errors.append(("schema", f"unknown task tag {tag!r}"))
    for s in ir.substances:
        if registry.resolve(s) is None:
            errors.append(("registry", f"unresolvable substance/entity {s!r}"))
    try:
        pred = Predicate(ir.predicate, registry)
        errors.extend(("predicate", m) for m in pred.vocabulary_errors())
    except PredicateError as exc:
        errors.append(("predicate", str(exc)))
    return errors


def is_valid_ir(ir: LabGuardIR, registry: Registry | None = None) -> bool:
    return not validate_ir(ir, registry)
