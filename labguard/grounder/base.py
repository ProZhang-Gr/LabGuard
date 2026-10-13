"""Grounder interface: natural-language rule -> LabGuard-IR (Sec. 3.4).

Every grounder returns a `LabGuardIR` for every input, even when the output is
malformed: missing fields become empty values so that the compiler (and the
grounding metrics) can reject or penalise them instead of the grounder
silently dropping rules.  Raw model outputs are kept in `last_raw` for
inspection.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import LabGuardIR, sort_tags, validate_ir

if TYPE_CHECKING:
    from labguard.bench.example import BenchExample

EMPTY_IR = LabGuardIR("", (), "", (), "", "")


class Grounder:
    """Base class.  Subclasses implement `ground` (and optionally `ground_batch`, `fit`)."""

    name = "base"
    trainable = False

    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or default_registry()
        self.last_raw: list[str] = []

    def ground(self, text: str) -> LabGuardIR:
        raise NotImplementedError

    def ground_batch(self, texts: Sequence[str]) -> list[LabGuardIR]:
        return [self.ground(t) for t in texts]

    def fit(self, train: Sequence["BenchExample"], dev: Sequence["BenchExample"] | None = None) -> "Grounder":
        """Trainable grounders learn from (rule, IR) pairs; the default is a no-op."""
        return self

    def validate(self, ir: LabGuardIR) -> list[tuple[str, str]]:
        return validate_ir(ir, self.registry)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.describe()})"


class OracleGrounder(Grounder):
    """Looks rule text up in a gold mapping (upper bound / tests).  Unknown text -> EMPTY_IR."""

    name = "oracle"

    def __init__(self, mapping: dict[str, LabGuardIR] | Iterable[tuple[str, LabGuardIR]],
                 registry: Registry | None = None) -> None:
        super().__init__(registry)
        self.mapping = {_key(k): v for k, v in (mapping.items() if isinstance(mapping, dict) else mapping)}

    @classmethod
    def from_examples(cls, examples: Iterable["BenchExample"], registry: Registry | None = None) -> "OracleGrounder":
        return cls(((e.text, e.ir) for e in examples), registry)

    def fit(self, train: Sequence["BenchExample"], dev: Sequence["BenchExample"] | None = None) -> "Grounder":
        for e in [*train, *(dev or [])]:
            self.mapping[_key(e.text)] = e.ir
        return self

    def ground(self, text: str) -> LabGuardIR:
        return self.mapping.get(_key(text), EMPTY_IR)


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def clean_ir(ir: LabGuardIR, registry: Registry | None = None) -> LabGuardIR:
    """Light, lossless clean-up applied to every model output (strip / lowercase / dedupe).

    It does not repair content: unknown values stay unknown so that Schema
    Valid / Compile Ready still measure the grounder itself.
    """
    registry = registry or default_registry()
    subs: list[str] = []
    for s in ir.substances:
        s = s.strip().strip("'\"")
        if s and s not in subs:
            subs.append(s)
    return replace(
        ir,
        constraint_type=ir.constraint_type.strip().lower().replace(" ", "_"),
        substances=tuple(subs),
        predicate=ir.predicate.strip(),
        action_scope=sort_tags(t.strip().lower() for t in ir.action_scope if t.strip()),
        severity=ir.severity.strip().lower(),
        intervention=ir.intervention.strip().lower(),
    )
