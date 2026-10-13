"""On-disk format of LabGuard-Bench (data/labguard_bench/).

  seeds.jsonl       203 seed rules (text, gold IR, source, difficulty, alternative wordings)
  examples.jsonl    812 labeled examples (seed wordings + reviewed paraphrases)
  splits.json       random / source-held-out / chemistry-held-out splits (example ids)
  stats.json        corpus statistics and comparison against the published counts
  annotation.json   (simulated) annotator-agreement report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from labguard.bench.example import BenchExample, SeedRule
from labguard.bench.splits import Split, all_splits, load_splits, save_splits

DEFAULT_BENCH_DIR = Path(__file__).resolve().parents[2] / "data" / "labguard_bench"


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_bench(bench: Any, out_dir: str | Path = DEFAULT_BENCH_DIR, split_seed: int = 42) -> dict[str, Path]:
    """Writes a `labguard.bench.synth.Bench` and its splits; returns the written paths."""
    from labguard.bench.synth import check_against_paper, config_dict

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / fn for name, fn in (("seeds", "seeds.jsonl"), ("examples", "examples.jsonl"),
                                               ("splits", "splits.json"), ("stats", "stats.json"),
                                               ("annotation", "annotation.json"))}
    write_jsonl((s.to_dict() for s in bench.seeds), paths["seeds"])
    write_jsonl((e.to_dict() for e in bench.examples), paths["examples"])
    save_splits(all_splits(bench.examples, split_seed, bench.config.held_out_chemical), paths["splits"])
    stats = bench.stats()
    stats["config"] = config_dict(bench.config)
    stats["paper_check"] = check_against_paper(stats)
    paths["stats"].write_text(json.dumps(stats, indent=1), encoding="utf-8")
    paths["annotation"].write_text(json.dumps(bench.annotation, indent=1), encoding="utf-8")
    return paths


def load_examples(path: str | Path = DEFAULT_BENCH_DIR) -> list[BenchExample]:
    path = Path(path)
    if path.is_dir():
        path = path / "examples.jsonl"
    return [BenchExample.from_dict(d) for d in read_jsonl(path)]


def load_seeds(path: str | Path = DEFAULT_BENCH_DIR) -> list[SeedRule]:
    path = Path(path)
    if path.is_dir():
        path = path / "seeds.jsonl"
    return [SeedRule.from_dict(d) for d in read_jsonl(path)]


def load_split(name: str, bench_dir: str | Path = DEFAULT_BENCH_DIR) -> tuple[list[BenchExample], list[BenchExample]]:
    """(train, test) examples of a saved split, e.g. "random", "source_held_out:expert"."""
    bench_dir = Path(bench_dir)
    splits = load_splits(bench_dir / "splits.json")
    if name not in splits:
        raise KeyError(f"unknown split {name!r}; available: {sorted(splits)}")
    return splits[name].select(load_examples(bench_dir))


def load_all_splits(bench_dir: str | Path = DEFAULT_BENCH_DIR) -> dict[str, Split]:
    return load_splits(Path(bench_dir) / "splits.json")


def ensure_bench(bench_dir: str | Path = DEFAULT_BENCH_DIR, seed: int = 0) -> Path:
    """Builds and saves the default corpus if it is not on disk yet."""
    bench_dir = Path(bench_dir)
    if not (bench_dir / "examples.jsonl").exists():
        from labguard.bench.synth import BenchConfig, build_bench

        save_bench(build_bench(BenchConfig(seed=seed)), bench_dir)
    return bench_dir
