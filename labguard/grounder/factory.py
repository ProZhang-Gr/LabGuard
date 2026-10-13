"""Builds grounders from short specs or config dicts / YAML files.

  regex                          rule-based baseline
  knn                            TF-IDF nearest-neighbour grounder (CPU, needs fit())
  llm:<model>                    OpenAI-compatible chat model (zero-shot; base_url via OPENAI_BASE_URL)
  hf:<model>                     local untuned causal LM, chat prompt (Base Grounder)
  lora:<adapter_dir>             LoRA-tuned grounder (base model read from the adapter)
  lora+cd:<adapter_dir>          ... with schema-constrained decoding
  scibert[:<checkpoint_dir>]     encoder classification baseline
  hybrid:<inner spec>            rule-based normalisation + regex substance fallback around <inner>
  <file>.yaml / <file>.json      config dict, e.g. {type: hybrid, neural: {type: lora, adapter: ..., constrained: true}}

Dict configs use the same `type` names; other keys are passed to the constructor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from labguard.grounder.base import Grounder
from labguard.ir.registry import Registry


def _load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("grounder", data)


def parse_spec(spec: str) -> dict[str, Any]:
    spec = spec.strip()
    if spec.endswith((".yaml", ".yml", ".json")) and Path(spec).exists():
        return _load_config(spec)
    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind == "hybrid":
        return {"type": "hybrid", "neural": parse_spec(arg) if arg else {"type": "regex"}}
    if kind in ("lora", "lora+cd"):
        return {"type": "lora", "adapter": arg or None, "constrained": kind == "lora+cd"}
    if kind in ("hf", "hf+cd"):
        return {"type": "hf", "model": arg, "constrained": kind == "hf+cd"}
    if kind == "llm":
        return {"type": "llm", "model": arg}
    if kind == "scibert":
        return {"type": "scibert", "checkpoint": arg or None}
    if kind in ("regex", "knn", "oracle") and not arg:
        return {"type": kind}
    raise ValueError(f"unknown grounder spec {spec!r}")


def build_grounder(spec: str | dict[str, Any], registry: Registry | None = None, **overrides: Any) -> Grounder:
    cfg = parse_spec(spec) if isinstance(spec, str) else dict(spec)
    cfg.update(overrides)
    kind = str(cfg.pop("type", "regex")).lower()
    if kind == "regex":
        from labguard.grounder.regex_grounder import RegexGrounder

        return RegexGrounder(registry)
    if kind == "knn":
        from labguard.grounder.retrieval import KNNGrounder

        return KNNGrounder(registry, **cfg)
    if kind == "oracle":
        from labguard.grounder.base import OracleGrounder

        return OracleGrounder(cfg.pop("mapping", {}), registry)
    if kind == "hybrid":
        from labguard.grounder.hybrid import HybridGrounder

        inner = cfg.pop("neural", {"type": "regex"})
        neural = build_grounder(inner if isinstance(inner, (dict, str)) else {"type": "regex"}, registry)
        return HybridGrounder(neural, registry, **cfg)
    if kind == "lora":
        from labguard.grounder.lora import LoRAGrounder

        return LoRAGrounder(registry=registry, **cfg)
    if kind == "hf":
        from labguard.grounder.hf import HFGrounder

        return HFGrounder(registry=registry, **cfg)
    if kind == "llm":
        from labguard.grounder.llm_grounder import ChatLLMGrounder

        return ChatLLMGrounder(registry=registry, **cfg)
    if kind in ("scibert", "encoder"):
        from labguard.grounder.encoder_baseline import EncoderGrounder

        return EncoderGrounder(registry=registry, **cfg)
    raise ValueError(f"unknown grounder type {kind!r}")
