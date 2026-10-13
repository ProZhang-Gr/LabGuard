"""Evaluation splits of LabGuard-Bench (Sec. 5.1, Table 6).

  random              seed-42 80/20 example-level split (649 / 163)
  source_held_out     leave-one-source-family-out, 4 folds; all seeds *and*
                      paraphrases of the held-out family are removed from training
  chemistry_held_out  every example whose label or text involves the held-out
                      chemical (HCl) is removed from training; the test set is
                      the 9 HCl seed rules
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from labguard.bench.example import BenchExample
from labguard.ir.registry import Registry, default_registry
from labguard.vocab import SOURCE_FAMILIES


@dataclass
class Split:
    name: str
    train: list[str]
    test: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def select(self, examples: Iterable[BenchExample]) -> tuple[list[BenchExample], list[BenchExample]]:
        by_id = {e.id: e for e in examples}
        return [by_id[i] for i in self.train], [by_id[i] for i in self.test]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Split":
        return cls(d["name"], list(d["train"]), list(d["test"]), dict(d.get("meta", {})))


def random_split(examples: Sequence[BenchExample], seed: int = 42, test_frac: float = 0.2) -> Split:
    ids = [e.id for e in examples]
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_test = math.ceil(len(ids) * test_frac)
    test = sorted((ids[i] for i in perm[:n_test]), key=ids.index)
    test_set = set(test)
    train = [i for i in ids if i not in test_set]
    return Split("random", train, test, {"seed": seed, "test_frac": test_frac})


def source_held_out_splits(examples: Sequence[BenchExample]) -> list[Split]:
    out = []
    for src in SOURCE_FAMILIES:
        train = [e.id for e in examples if e.source != src]
        test = [e.id for e in examples if e.source == src]
        out.append(Split(f"source_held_out:{src}", train, test, {"held_out_source": src}))
    return out


def involves(example: BenchExample, chemical: str, registry: Registry | None = None) -> bool:
    registry = registry or default_registry()
    return chemical in example.ir.substances or any(sid == chemical for sid, _ in registry.find_mentions(example.text))


def chemistry_held_out_split(examples: Sequence[BenchExample], chemical: str = "HCl", test_on: str = "seeds",
                             registry: Registry | None = None) -> Split:
    """`test_on="seeds"` evaluates on the seed wordings (N = 9 in the paper); "all" adds paraphrases."""
    registry = registry or default_registry()
    held = [e for e in examples if involves(e, chemical, registry)]
    held_ids = {e.id for e in held}
    train = [e.id for e in examples if e.id not in held_ids]
    test = [e.id for e in held if test_on == "all" or not e.is_paraphrase]
    return Split(f"chemistry_held_out:{chemical}", train, test, {"chemical": chemical, "test_on": test_on})


def all_splits(examples: Sequence[BenchExample], seed: int = 42, chemical: str = "HCl") -> dict[str, Split]:
    splits = {"random": random_split(examples, seed)}
    for s in source_held_out_splits(examples):
        splits[s.name] = s
    ch = chemistry_held_out_split(examples, chemical)
    splits[ch.name] = ch
    return splits


def save_splits(splits: dict[str, Split], path: str | Path) -> None:
    Path(path).write_text(json.dumps({k: v.to_dict() for k, v in splits.items()}, indent=1), encoding="utf-8")


def load_splits(path: str | Path) -> dict[str, Split]:
    return {k: Split.from_dict(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}
