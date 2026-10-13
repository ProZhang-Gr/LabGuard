"""Builds the LabGuard-Bench reconstruction: 203 seed rules -> 812 labeled examples.

Published statistics that the build reproduces exactly:
  * 812 examples = 203 seeds x (1 seed wording + 3 reviewed paraphrases)
  * per-type counts of Table 7 (seed counts = Table 7 / 4)
  * per-source seed counts 71 / 62 / 50 / 20 (lab manual / PubChem GHS /
    expert / chemical-safety reference), with type mixes following Table 8
  * difficulty strata easy 71 / medium 105 / hard 27 (by structural
    complexity rank of the seed)
  * exactly 9 seed rules involving hydrochloric acid (chemistry-held-out
    test set, Table 2)

The released benchmark text is not public, so the corpus is synthetic; see
`templates.py` for the wording model.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from labguard.bench.annotation import AnnotatorModel, adjudicate, agreement_report, simulate_annotators
from labguard.bench.example import BenchExample, SeedRule
from labguard.bench.paraphrase import Paraphraser, build_paraphraser
from labguard.bench.templates import NoCandidate, RuleDraft, generate_rule
from labguard.compiler.compiler import Compiler
from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import validate_ir
from labguard.ir.serialize import ir_to_text
from labguard.vocab import (
    CONSTRAINT_TYPES,
    N_PARAPHRASES_PER_SEED,
    PAPER_DIFFICULTY_COUNTS,
    PAPER_SOURCE_COUNTS,
    PAPER_TYPE_COUNTS,
    SOURCE_FAMILIES,
)

# Seed rules per (source family, constraint type).  Row sums = per-source
# seed counts, column sums = Table 7 / 4; the dominant families per source
# follow Table 8.
SOURCE_TYPE_ALLOCATION: dict[str, dict[str, int]] = {
    "lab_manual": {
        "equipment_check": 13, "sequence_constraint": 10, "temperature_constraint": 10, "velocity_limit": 10,
        "spill_prevention": 9, "container_capacity": 6, "workspace_zone": 5, "force_limit": 5,
        "waste_handling": 3,
    },
    "pubchem_ghs": {
        "exposure_limit": 17, "ventilation_requirement": 15, "fire_safety": 14, "ppe_requirement": 13,
        "waste_handling": 3,
    },
    "expert": {
        "chemical_incompatibility": 16, "equipment_check": 12, "sequence_constraint": 10,
        "temperature_constraint": 6, "storage_constraint": 3, "waste_handling": 3,
    },
    "chem_safety_ref": {"chemical_incompatibility": 17, "storage_constraint": 3},
}

# Seeds that must involve the held-out chemical (chemistry-held-out split, N = 9)
HELD_OUT_QUOTA: dict[tuple[str, str], int] = {
    ("chem_safety_ref", "chemical_incompatibility"): 2,
    ("expert", "chemical_incompatibility"): 1,
    ("expert", "storage_constraint"): 1,
    ("pubchem_ghs", "ventilation_requirement"): 1,
    ("pubchem_ghs", "exposure_limit"): 1,
    ("pubchem_ghs", "ppe_requirement"): 1,
    ("lab_manual", "spill_prevention"): 1,
    ("lab_manual", "waste_handling"): 1,
}
SOURCE_PREFIX = {"lab_manual": "LM", "pubchem_ghs": "GHS", "expert": "EX", "chem_safety_ref": "CSR"}


@dataclass
class BenchConfig:
    seed: int = 0
    n_paraphrases: int = N_PARAPHRASES_PER_SEED
    held_out_chemical: str = "HCl"
    paraphraser: str = "template"
    annotate: bool = True
    n_annotators: int = 5
    max_tries: int = 400
    difficulty_counts: dict[str, int] = field(default_factory=lambda: dict(PAPER_DIFFICULTY_COUNTS))


@dataclass
class Bench:
    seeds: list[SeedRule]
    examples: list[BenchExample]
    config: BenchConfig
    annotation: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def example(self, eid: str) -> BenchExample:
        for e in self.examples:
            if e.id == eid:
                return e
        raise KeyError(eid)

    def stats(self) -> dict[str, Any]:
        return bench_stats(self.seeds, self.examples, self.config.held_out_chemical)


def _allocation_slots() -> list[tuple[str, str, bool]]:
    slots = []
    for source in SOURCE_FAMILIES:
        for ctype, n in SOURCE_TYPE_ALLOCATION[source].items():
            q = HELD_OUT_QUOTA.get((source, ctype), 0)
            slots += [(source, ctype, k < q) for k in range(n)]
    return slots


def _texts_ok(texts: list[str], chem: str, must_mention: bool, registry: Registry) -> list[str]:
    """Wordings that mention the held-out chemical iff the label involves it."""
    return [t for t in dict.fromkeys(texts)
            if any(sid == chem for sid, _ in registry.find_mentions(t)) == must_mention]


def _generate_seed(source: str, ctype: str, needs_chem: bool, chem: str, rng: np.random.Generator,
                   registry: Registry, seen_ir: set[str], seen_text: set[str], max_tries: int) -> RuleDraft:
    require = chem if needs_chem else None
    exclude = () if needs_chem else (chem,)
    fallback: RuleDraft | None = None
    for attempt in range(max_tries):
        try:
            d = generate_rule(ctype, source, rng, registry, require=require, exclude=exclude)
        except NoCandidate:
            continue
        if (chem in d.ir.substances) != needs_chem:
            continue
        texts = _texts_ok(d.texts, chem, needs_chem, registry)
        texts = [t for t in texts if t not in seen_text]
        if not texts:
            continue
        d.texts = texts
        key = ir_to_text(d.ir.normalized(registry))
        if key in seen_ir:
            fallback = fallback or d
            continue
        seen_ir.add(key)
        return d
    if fallback is None:
        raise RuntimeError(f"cannot generate a seed for {source}/{ctype} (require={require})")
    fallback.flags["duplicate_ir"] = True  # different wording of an already-sampled rule
    return fallback


def assign_difficulty(seeds: list[SeedRule], counts: dict[str, int], rng: np.random.Generator) -> None:
    """Difficulty strata by structural-complexity rank (ties broken at random)."""
    order = sorted(range(len(seeds)), key=lambda i: (seeds[i].complexity, rng.random()))
    total = sum(counts.values())
    bounds, acc = [], 0
    for level in ("easy", "medium", "hard"):
        acc += counts[level]
        bounds.append((level, round(acc * len(seeds) / total)))
    k = 0
    for rank, i in enumerate(order):
        while rank >= bounds[k][1]:
            k += 1
        seeds[i].difficulty = bounds[k][0]


def generate_seeds(config: BenchConfig, registry: Registry | None = None) -> list[SeedRule]:
    registry = registry or default_registry()
    rng = np.random.default_rng(config.seed)
    compiler = Compiler(registry)
    seen_ir: set[str] = set()
    seen_text: set[str] = set()
    seeds: list[SeedRule] = []
    counters: Counter[str] = Counter()
    for source, ctype, needs_chem in _allocation_slots():
        d = _generate_seed(source, ctype, needs_chem, config.held_out_chemical, rng, registry, seen_ir, seen_text,
                           config.max_tries)
        errors = validate_ir(d.ir, registry)
        if errors:
            raise AssertionError(f"generator produced an invalid IR {d.ir}: {errors}")
        compiler.compile(d.ir)  # raises if the gold label cannot be lowered
        counters[source] += 1
        sid = f"{SOURCE_PREFIX[source]}{counters[source]:03d}"
        seen_text.update(d.texts)
        seeds.append(SeedRule(id=sid, text=d.texts[0], ir=d.ir.with_meta(sid, d.texts[0]), source=source,
                              complexity=round(float(d.complexity), 3), template=d.template,
                              alt_texts=d.texts[1:], flags=d.flags))
    assign_difficulty(seeds, config.difficulty_counts, rng)
    return seeds


def expand_examples(seeds: list[SeedRule], config: BenchConfig, paraphraser: Paraphraser | None = None,
                    registry: Registry | None = None) -> list[BenchExample]:
    registry = registry or default_registry()
    paraphraser = paraphraser or build_paraphraser(config.paraphraser, registry=registry)
    rng = np.random.default_rng(config.seed + 1)
    examples: list[BenchExample] = []
    for s in seeds:
        meta = {"template": s.template, "complexity": s.complexity}
        examples.append(BenchExample(f"{s.id}-0", s.id, s.text, s.ir.with_meta(f"{s.id}-0", s.text), s.source,
                                     s.difficulty, False, None, None, dict(meta)))
        paras = paraphraser.paraphrase(s.text, s.alt_texts, s.ir.severity, config.n_paraphrases, rng, registry)
        if len(paras) < config.n_paraphrases:
            raise RuntimeError(f"only {len(paras)} reviewed paraphrases for seed {s.id}: {s.text!r}")
        for j, (text, method) in enumerate(paras, start=1):
            eid = f"{s.id}-{j}"
            examples.append(BenchExample(eid, s.id, text, s.ir.with_meta(eid, text), s.source, s.difficulty, True,
                                         method, None, dict(meta)))
    return examples


def annotate_examples(examples: list[BenchExample], config: BenchConfig, registry: Registry | None = None,
                      model: AnnotatorModel | None = None) -> dict[str, Any]:
    registry = registry or default_registry()
    rng = np.random.default_rng(config.seed + 2)
    results = []
    for e in examples:
        labels = simulate_annotators(e.ir, e.difficulty, e.is_paraphrase, rng, config.n_annotators, model, registry)
        adj = adjudicate(labels, draft=e.ir)
        e.agreement = round(adj.agreement, 4)
        e.meta["revised"] = adj.revised
        results.append(adj)
    report = agreement_report(results)
    report["simulated"] = True
    return report


def build_bench(config: BenchConfig | None = None, registry: Registry | None = None,
                paraphraser: Paraphraser | None = None) -> Bench:
    config = config or BenchConfig()
    registry = registry or default_registry()
    seeds = generate_seeds(config, registry)
    examples = expand_examples(seeds, config, paraphraser, registry)
    annotation = annotate_examples(examples, config, registry) if config.annotate else {}
    return Bench(seeds, examples, config, annotation)


def bench_stats(seeds: list[SeedRule], examples: list[BenchExample], held_out_chemical: str = "HCl",
                registry: Registry | None = None) -> dict[str, Any]:
    registry = registry or default_registry()
    compiler = Compiler(registry)
    type_counts = Counter(e.constraint_type for e in examples)
    src_types: dict[str, dict[str, int]] = {}
    for s in seeds:
        src_types.setdefault(s.source, Counter())[s.constraint_type] += 1  # type: ignore[index]
    return {
        "n_seeds": len(seeds),
        "n_examples": len(examples),
        "type_counts": {t: type_counts.get(t, 0) for t in CONSTRAINT_TYPES},
        "source_seed_counts": dict(Counter(s.source for s in seeds)),
        "difficulty_seed_counts": dict(Counter(s.difficulty for s in seeds)),
        "source_type_seed_counts": {k: dict(v) for k, v in src_types.items()},
        "held_out_chemical": held_out_chemical,
        "held_out_seeds": sum(held_out_chemical in s.ir.substances for s in seeds),
        "held_out_examples": sum(held_out_chemical in e.ir.substances for e in examples),
        "paraphrase_methods": dict(Counter(e.paraphrase_method for e in examples if e.is_paraphrase)),
        "severity_counts": dict(Counter(s.ir.severity for s in seeds)),
        "intervention_counts": dict(Counter(s.ir.intervention for s in seeds)),
        "unique_seed_irs": len({ir_to_text(s.ir.normalized(registry)) for s in seeds}),
        "schema_valid": float(np.mean([not validate_ir(s.ir, registry) for s in seeds])),
        "compile_ready": float(np.mean([compiler.try_compile(s.ir) is not None for s in seeds])),
        "mean_complexity": {d: round(float(np.mean([s.complexity for s in seeds if s.difficulty == d])), 3)
                            for d in ("easy", "medium", "hard")},
        "mean_tokens": round(float(np.mean([len(e.text.split()) for e in examples])), 2),
    }


def check_against_paper(stats: dict[str, Any]) -> list[str]:
    """Discrepancies between a built corpus and the published statistics (empty list = match)."""
    issues = []
    if stats["n_examples"] != sum(PAPER_TYPE_COUNTS.values()):
        issues.append(f"n_examples {stats['n_examples']} != 812")
    for t, n in PAPER_TYPE_COUNTS.items():
        if stats["type_counts"].get(t) != n:
            issues.append(f"type {t}: {stats['type_counts'].get(t)} != {n}")
    for s, n in PAPER_SOURCE_COUNTS.items():
        if stats["source_seed_counts"].get(s) != n:
            issues.append(f"source {s}: {stats['source_seed_counts'].get(s)} != {n}")
    for d, n in PAPER_DIFFICULTY_COUNTS.items():
        if stats["difficulty_seed_counts"].get(d) != n:
            issues.append(f"difficulty {d}: {stats['difficulty_seed_counts'].get(d)} != {n}")
    if stats["held_out_seeds"] != 9:
        issues.append(f"held-out chemical seeds {stats['held_out_seeds']} != 9")
    if stats["schema_valid"] < 1.0 or stats["compile_ready"] < 1.0:
        issues.append("gold labels must be schema-valid and compile-ready")
    return issues


def config_dict(config: BenchConfig) -> dict[str, Any]:
    return asdict(config)


def expected_split_sizes(n_examples: int = 812, test_frac: float = 0.2) -> tuple[int, int]:
    n_test = math.ceil(n_examples * test_frac)
    return n_examples - n_test, n_test
