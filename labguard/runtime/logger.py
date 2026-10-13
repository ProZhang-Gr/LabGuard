"""Structured safety log (one record per control step)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return np.round(x, 5).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    return x


@dataclass
class SafetyLogger:
    enabled: bool = True
    records: list[dict[str, Any]] = field(default_factory=list)

    def log(self, **record: Any) -> None:
        if self.enabled:
            self.records.append(_jsonable(record))

    def clear(self) -> None:
        self.records.clear()

    def counts(self) -> dict[str, int]:
        c = Counter(r.get("verdict", "pass") for r in self.records)
        return {"steps": len(self.records), "pass": c["pass"], "modify": c["modify"], "block": c["block"],
                "high_risk": sum(1 for r in self.records if r.get("high_risk"))}

    def rule_counts(self) -> dict[str, int]:
        c: Counter[str] = Counter()
        for r in self.records:
            for f in r.get("fired", []):
                c[f.get("rule_id") or f.get("monitor", "?")] += 1
        return dict(c)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in self.records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
