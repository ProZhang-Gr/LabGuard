"""Retrieval utilities and a CPU-only nearest-neighbour grounder.

`KNNGrounder` is not one of the paper's baselines; it is a cheap trainable
grounder that lets the full pipeline (training split -> grounding ->
compilation -> replay) run without a GPU.  It copies the IR of the most
similar training rule (character n-gram TF-IDF) and re-binds the
rule-specific slots from the new text: substances via the registry scanner
and numeric thresholds via unit-aware number extraction.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from labguard.grounder.base import EMPTY_IR, Grounder
from labguard.grounder.regex_grounder import extract_quantities, extract_substances, fmt_num
from labguard.ir.registry import Registry
from labguard.ir.schema import LabGuardIR

if TYPE_CHECKING:
    from labguard.bench.example import BenchExample

_PRED_NUM = re.compile(r"(?<![\w.'])(\d+(?:\.\d+)?)(?![\w'])")


class TfidfIndex:
    def __init__(self, texts: Sequence[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True)
        self.mat = self.vec.fit_transform(list(texts))

    def scores(self, text: str) -> np.ndarray:
        return (self.mat @ self.vec.transform([text]).T).toarray().ravel()

    def nearest(self, text: str, k: int = 1) -> list[int]:
        s = self.scores(text)
        return [int(i) for i in np.argsort(-s)[:k]]


def rebind_numbers(pred: str, text: str) -> str:
    """Replace the numeric constants of a predicate with the numbers of `text`, in order."""
    nums = [fmt_num(q.value) for q in extract_quantities(text)]
    if not nums:
        return pred
    it = iter(nums)
    return _PRED_NUM.sub(lambda m: next(it, m.group(1)), pred)


class KNNGrounder(Grounder):
    name = "knn"
    trainable = True

    def __init__(self, registry: Registry | None = None, rebind: bool = True) -> None:
        super().__init__(registry)
        self.rebind = rebind
        self._train: list["BenchExample"] = []
        self._index: TfidfIndex | None = None

    def fit(self, train: Sequence["BenchExample"], dev: Sequence["BenchExample"] | None = None) -> "KNNGrounder":
        self._train = list(train)
        self._index = TfidfIndex([e.text for e in self._train])
        return self

    def ground(self, text: str) -> LabGuardIR:
        if self._index is None:
            return EMPTY_IR
        ir = self._train[self._index.nearest(text, 1)[0]].ir
        if not self.rebind:
            return ir
        subs = extract_substances(text, self.registry, ir.constraint_type, ir.predicate)
        return replace(ir, substances=subs, predicate=rebind_numbers(ir.predicate, text), rule_id=None,
                       rule_text=None)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "n_train": len(self._train), "rebind": self.rebind}
