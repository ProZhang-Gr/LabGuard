"""Runtime rule sets: natural-language rules + optional gold IR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import yaml

from labguard.ir.schema import LabGuardIR

if TYPE_CHECKING:
    from labguard.grounder.base import Grounder

DEFAULT_RULES_PATH = Path(__file__).with_name("default_rules.yaml")


@dataclass(frozen=True)
class RuntimeRule:
    rule_id: str
    text: str
    gold: LabGuardIR | None = None


def load_rules(path: str | Path | None = None) -> list[RuntimeRule]:
    path = Path(path) if path else DEFAULT_RULES_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for r in data["rules"]:
        gold = None
        if r.get("gold"):
            gold = LabGuardIR.from_dict(r["gold"]).with_meta(rule_id=r["id"], rule_text=r["text"])
        out.append(RuntimeRule(r["id"], r["text"], gold))
    return out


def gold_irs(rules: Iterable[RuntimeRule]) -> list[LabGuardIR]:
    return [r.gold for r in rules if r.gold is not None]


def ground_rules(rules: Iterable[RuntimeRule], grounder: "Grounder") -> list[LabGuardIR]:
    rules = list(rules)
    preds = grounder.ground_batch([r.text for r in rules])
    return [p.with_meta(rule_id=r.rule_id, rule_text=r.text) for p, r in zip(preds, rules)]
