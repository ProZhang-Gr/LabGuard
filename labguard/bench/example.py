"""Data records of LabGuard-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from labguard.ir.schema import LabGuardIR
from labguard.vocab import TYPE_TO_FAMILY


@dataclass
class SeedRule:
    """One of the 203 seed rules: a source-style wording and its reviewed gold IR."""

    id: str
    text: str
    ir: LabGuardIR
    source: str
    difficulty: str = "medium"
    complexity: float = 0.0
    template: str = ""
    alt_texts: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def constraint_type(self) -> str:
        return self.ir.constraint_type

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "ir": self.ir.to_dict(), "source": self.source,
                "difficulty": self.difficulty, "complexity": self.complexity, "template": self.template,
                "alt_texts": list(self.alt_texts), "flags": dict(self.flags)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SeedRule":
        return cls(id=d["id"], text=d["text"], ir=LabGuardIR.from_dict(d["ir"]).with_meta(d["id"], d["text"]),
                   source=d["source"], difficulty=d.get("difficulty", "medium"),
                   complexity=float(d.get("complexity", 0.0)), template=d.get("template", ""),
                   alt_texts=list(d.get("alt_texts", [])), flags=dict(d.get("flags", {})))


@dataclass
class BenchExample:
    """A labeled (rule text, gold IR) pair; paraphrases inherit the seed label after review."""

    id: str
    seed_id: str
    text: str
    ir: LabGuardIR
    source: str
    difficulty: str
    is_paraphrase: bool = False
    paraphrase_method: str | None = None
    agreement: float | None = None  # fraction of annotator votes matching the final label
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def constraint_type(self) -> str:
        return self.ir.constraint_type

    @property
    def family(self) -> str:
        return TYPE_TO_FAMILY.get(self.ir.constraint_type, "unknown")

    @property
    def gold(self) -> LabGuardIR:
        return self.ir.with_meta(self.id, self.text)

    def mentions(self, sid: str) -> bool:
        return sid in self.ir.substances

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "seed_id": self.seed_id, "text": self.text, "ir": self.ir.to_dict(),
                "source": self.source, "difficulty": self.difficulty, "is_paraphrase": self.is_paraphrase,
                "paraphrase_method": self.paraphrase_method, "agreement": self.agreement,
                "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchExample":
        return cls(id=d["id"], seed_id=d["seed_id"], text=d["text"],
                   ir=LabGuardIR.from_dict(d["ir"]).with_meta(d["id"], d["text"]), source=d["source"],
                   difficulty=d.get("difficulty", "medium"), is_paraphrase=bool(d.get("is_paraphrase", False)),
                   paraphrase_method=d.get("paraphrase_method"), agreement=d.get("agreement"),
                   meta=dict(d.get("meta", {})))
