"""Grounding evaluation protocols (Sec. 5.1-5.2, Table 2, Table 6).

  random      seed-42 80/20 split (649 / 163 examples), upper-bound setting
  source      leave-one-source-out, 4 folds, mean±std over folds
  chemistry   all HCl rules held out (9 seed rules), mean±std over 5 training seeds

A grounder is supplied as a *builder* `build(split_name, seed) -> Grounder`
so that trainable grounders get a fresh model (and a fold-specific adapter
directory) per fold and seed.  `make_builder` turns a factory spec such as
"regex", "knn", "lora:<root>", "hybrid:lora+cd:<root>" into a builder.
"""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from labguard.bench.example import BenchExample
from labguard.bench.io import DEFAULT_BENCH_DIR, load_all_splits, load_examples
from labguard.eval.metrics import GROUNDING_KEYS, aggregate, grounding_scores
from labguard.grounder.base import Grounder
from labguard.grounder.factory import build_grounder, parse_spec
from labguard.ir.registry import Registry
from labguard.ir.serialize import ir_to_text

Builder = Callable[[str, int], Grounder]

PROTOCOL_SEEDS: dict[str, tuple[int, ...]] = {"random": (0,), "source": (0,), "chemistry": (0, 1, 2, 3, 4)}


def protocol_splits(protocol: str, bench_dir: str | Path = DEFAULT_BENCH_DIR) -> list[str]:
    names = sorted(load_all_splits(bench_dir))
    if protocol == "random":
        return ["random"]
    if protocol == "source":
        return [n for n in names if n.startswith("source_held_out:")]
    if protocol == "chemistry":
        return [n for n in names if n.startswith("chemistry_held_out:")]
    if protocol in names:
        return [protocol]
    raise KeyError(f"unknown protocol / split {protocol!r}")


def dev_holdout(train: Sequence[BenchExample], frac: float, seed: int) -> tuple[list[BenchExample], list[BenchExample]]:
    """Hold out whole seed rules (with their paraphrases) for model selection."""
    if frac <= 0:
        return list(train), []
    seeds = sorted({e.seed_id for e in train})
    rng = random.Random(seed)
    rng.shuffle(seeds)
    dev_seeds = set(seeds[:max(1, int(round(frac * len(seeds))))])
    return [e for e in train if e.seed_id not in dev_seeds], [e for e in train if e.seed_id in dev_seeds]


def _with_root(cfg: dict[str, Any], root: str | None, split: str, seed: int) -> dict[str, Any]:
    """Point every LoRA / encoder node of a config tree at a fold-specific directory."""
    cfg = copy.deepcopy(cfg)
    kind = cfg.get("type")
    tag = split.replace(":", "_")
    if kind == "hybrid":
        cfg["neural"] = _with_root(cfg.get("neural", {"type": "regex"}), root, split, seed)
    elif kind == "lora":
        base = cfg.get("adapter") or root or "runs/lora"
        cfg["adapter"] = str(Path(base) / tag / f"seed{seed}")
        tc = dict(cfg.get("train_config") or {})
        tc["seed"] = seed
        cfg["train_config"] = tc
    elif kind in ("scibert", "encoder"):
        base = cfg.get("checkpoint") or root or "runs/scibert"
        cfg["checkpoint"] = str(Path(base) / tag / f"seed{seed}")
    return cfg


def make_builder(spec: str | dict[str, Any], registry: Registry | None = None, root: str | None = None,
                 **overrides: Any) -> Builder:
    base = parse_spec(spec) if isinstance(spec, str) else dict(spec)
    base.update(overrides)

    def build(split: str, seed: int) -> Grounder:
        return build_grounder(_with_root(base, root, split, seed), registry)

    return build


def _needs_fit(g: Grounder, reuse: bool) -> bool:
    if not g.trainable:
        return False
    if not reuse:
        return True
    inner = getattr(g, "neural", g)
    trained = getattr(inner, "adapter", None) or getattr(inner, "_net", None) is not None
    return not trained


def fit_grounder(spec: str | dict[str, Any] | Builder, split: str = "random", seed: int = 0,
                 bench_dir: str | Path = DEFAULT_BENCH_DIR, registry: Registry | None = None,
                 root: str | None = None, dev_frac: float = 0.1, reuse: bool = True) -> Grounder:
    """Build a grounder and fit it on the training part of `split` (no-op for rule-based grounders)."""
    build = spec if callable(spec) else make_builder(spec, registry, root)
    g = build(split, seed)
    if _needs_fit(g, reuse):
        train, _ = load_all_splits(bench_dir)[split].select(load_examples(bench_dir))
        tr, dev = dev_holdout(train, dev_frac, seed)
        g.fit(tr, dev or None)
    return g


def evaluate_grounder(build: Builder, split_names: Iterable[str], bench_dir: str | Path = DEFAULT_BENCH_DIR,
                      seeds: Sequence[int] = (0,), dev_frac: float = 0.1, registry: Registry | None = None,
                      reuse: bool = True, keep_predictions: bool = False,
                      on_trained: Callable[[str, int, Grounder], None] | None = None,
                      log: Callable[[str], None] | None = print) -> dict[str, Any]:
    bench_dir = Path(bench_dir)
    examples = load_examples(bench_dir)
    splits = load_all_splits(bench_dir)
    runs: list[dict[str, Any]] = []
    predictions: dict[str, list[dict[str, Any]]] = {}
    describe: dict[str, Any] = {}
    for name in split_names:
        train, test = splits[name].select(examples)
        for seed in seeds:
            g = build(name, seed)
            describe = g.describe()
            t0 = time.perf_counter()
            if _needs_fit(g, reuse):
                tr, dev = dev_holdout(train, dev_frac, seed)
                g.fit(tr, dev or None)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            preds = g.ground_batch([e.text for e in test])
            t_ground = time.perf_counter() - t0
            scores = grounding_scores(preds, [e.gold for e in test], registry)
            runs.append({"split": name, "seed": seed, "n_train": len(train), "fit_s": t_fit,
                         "ground_s": t_ground, **scores})
            if keep_predictions:
                raw = list(getattr(g, "last_raw", []) or [])
                predictions[f"{name}#{seed}"] = [
                    {"id": e.id, "text": e.text, "gold": ir_to_text(e.ir), "pred": ir_to_text(p),
                     "raw": raw[i] if i < len(raw) else None} for i, (e, p) in enumerate(zip(test, preds))]
            if on_trained is not None:
                on_trained(name, seed, g)
            if log:
                log(f"[{describe.get('name')}] {name} seed={seed} n={len(test)} "
                    + " ".join(f"{k}={scores[k]:.3f}" for k in ("type_acc", "subst_f1", "tag_f1", "pred_soft",
                                                                "pred_em", "compile_ready")))
    out: dict[str, Any] = {"grounder": describe, "runs": runs, "summary": aggregate(runs, GROUNDING_KEYS)}
    if keep_predictions:
        out["predictions"] = predictions
    return out


def run_protocol(spec: str | dict[str, Any] | Builder, protocol: str, bench_dir: str | Path = DEFAULT_BENCH_DIR,
                 seeds: Sequence[int] | None = None, registry: Registry | None = None, root: str | None = None,
                 out_path: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    build = spec if callable(spec) else make_builder(spec, registry, root)
    seeds = tuple(seeds) if seeds is not None else PROTOCOL_SEEDS.get(protocol, (0,))
    res = evaluate_grounder(build, protocol_splits(protocol, bench_dir), bench_dir, seeds, registry=registry,
                            **kwargs)
    res["protocol"] = protocol
    res["spec"] = spec if isinstance(spec, (str, dict)) else getattr(spec, "__name__", "builder")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return res
