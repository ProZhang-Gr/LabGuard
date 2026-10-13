"""LoRA fine-tuning of the grounder (Sec. 3.4, Eq. 3).

The loss is the token-level negative log-likelihood of the serialized gold IR
given the rule; prompt tokens are masked out (label -100).  The paper uses
Qwen3-8B; it does not report LoRA hyper-parameters, so the defaults below are
standard choices (r = 16, alpha = 32, all attention + MLP projections,
lr 2e-4, 3 epochs, effective batch 16).  On a 16 GB GPU use
`load_in_4bit=True` (QLoRA, needs bitsandbytes).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from labguard.grounder.hf import HFGrounder, default_device, load_causal_lm, load_tokenizer
from labguard.grounder.prompts import sft_prompt, sft_target

if TYPE_CHECKING:
    from labguard.bench.example import BenchExample

IGNORE_INDEX = -100


@dataclass
class LoRATrainConfig:
    base_model: str = "Qwen/Qwen3-8B"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_len: int = 384
    load_in_4bit: bool = False
    dtype: str = "auto"
    gradient_checkpointing: bool = True
    seed: int = 0
    log_every: int = 10
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "LoRATrainConfig":
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        if "target_modules" in known:
            known["target_modules"] = tuple(known["target_modules"])
        return cls(**known, extra=d)


def encode_example(tokenizer: Any, rule: str, target: str, max_len: int) -> dict[str, list[int]]:
    """input = prompt + target + eos; labels mask the prompt with -100."""
    p = tokenizer(sft_prompt(rule), add_special_tokens=False)["input_ids"]
    t = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    p = p[-max(1, max_len - len(t)):]  # keep the tail of over-long prompts
    ids = (p + t)[:max_len]
    labels = ([IGNORE_INDEX] * len(p) + t)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate(batch: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    import torch

    n = max(len(b["input_ids"]) for b in batch)

    def pad(key: str, value: int) -> Any:
        return torch.tensor([b[key] + [value] * (n - len(b[key])) for b in batch], dtype=torch.long)

    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", IGNORE_INDEX)}


def build_features(examples: Sequence["BenchExample"], tokenizer: Any, max_len: int) -> list[dict[str, list[int]]]:
    return [encode_example(tokenizer, e.text, sft_target(e.ir), max_len) for e in examples]


def _eval_loss(model: Any, feats: list[dict[str, list[int]]], pad_id: int, batch_size: int) -> float:
    import torch

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            b = {k: v.to(model.device) for k, v in collate(feats[i:i + batch_size], pad_id).items()}
            ntok = int((b["labels"][:, 1:] != IGNORE_INDEX).sum())
            tot += float(model(**b).loss) * ntok
            n += ntok
    model.train()
    return tot / max(1, n)


def train_lora(train: Sequence["BenchExample"], out_dir: str | Path, cfg: LoRATrainConfig | None = None,
               dev: Sequence["BenchExample"] | None = None, model: Any = None, tokenizer: Any = None,
               verbose: bool = True, return_model: bool = False) -> dict[str, Any]:
    """Fine-tunes a LoRA adapter on (rule, IR) pairs and saves it to `out_dir`.

    `model` / `tokenizer` may be passed directly (tests use a tiny randomly
    initialised model); otherwise `cfg.base_model` is loaded.  With
    `return_model=True` the returned dict also holds the trained "model" and
    "tokenizer" (with dev data, the adapter of the best dev epoch is reloaded).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_linear_schedule_with_warmup

    cfg = cfg or LoRATrainConfig()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = tokenizer or load_tokenizer(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = model or load_causal_lm(cfg.base_model, cfg.load_in_4bit, cfg.dtype)
    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable") and default_device() == "cuda":
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    present = {n.rsplit(".", 1)[-1] for n, _ in base.named_modules()}
    targets = [m for m in cfg.target_modules if m in present] or None
    peft_model = get_peft_model(base, LoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                                                 target_modules=targets, task_type="CAUSAL_LM"))
    if verbose:
        peft_model.print_trainable_parameters()

    feats = build_features(train, tok, cfg.max_len)
    dev_feats = build_features(dev, tok, cfg.max_len) if dev else []
    steps_per_epoch = math.ceil(len(feats) / (cfg.batch_size * cfg.grad_accum))
    total = max(1, steps_per_epoch * cfg.epochs)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
    history: list[dict[str, float]] = []
    best = math.inf
    step, t0 = 0, time.time()
    peft_model.train()
    for epoch in range(cfg.epochs):
        order = list(range(len(feats)))
        random.shuffle(order)
        running, nb = 0.0, 0
        for bi in range(0, len(order), cfg.batch_size):
            batch = collate([feats[i] for i in order[bi:bi + cfg.batch_size]], tok.pad_token_id)
            batch = {k: v.to(peft_model.device) for k, v in batch.items()}
            loss = peft_model(**batch).loss / cfg.grad_accum
            loss.backward()
            running += loss.item() * cfg.grad_accum
            nb += 1
            if nb % cfg.grad_accum == 0 or bi + cfg.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if verbose and step % cfg.log_every == 0:
                    print(f"[lora] epoch {epoch} step {step}/{total} loss {running / nb:.4f} "
                          f"({time.time() - t0:.0f}s)")
        rec = {"epoch": epoch, "train_loss": running / max(1, nb)}
        if dev_feats:
            rec["dev_loss"] = _eval_loss(peft_model, dev_feats, tok.pad_token_id, cfg.batch_size)
            if rec["dev_loss"] < best:
                best = rec["dev_loss"]
                peft_model.save_pretrained(out_dir)
        history.append(rec)
        if verbose:
            print(f"[lora] epoch {epoch} done: {rec}")
    if not dev_feats:
        peft_model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    meta = {"base_model": cfg.base_model, "config": asdict(cfg), "history": history, "n_train": len(feats),
            "n_dev": len(dev_feats), "seconds": round(time.time() - t0, 1)}
    (out_dir / "labguard_lora.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if return_model:
        if dev_feats and len(history) > 1:
            from peft import set_peft_model_state_dict
            from peft.utils import load_peft_weights

            set_peft_model_state_dict(peft_model, load_peft_weights(str(out_dir), device=str(peft_model.device)))
        peft_model.eval()
        return {**meta, "model": peft_model, "tokenizer": tok}
    return meta


class LoRAGrounder(HFGrounder):
    """HFGrounder that can also train its own adapter (`fit`)."""

    trainable = True

    def __init__(self, adapter: str | None = None, base_model: str | None = None,
                 train_config: LoRATrainConfig | dict[str, Any] | None = None, **kwargs: Any) -> None:
        cfg = train_config if isinstance(train_config, LoRATrainConfig) else LoRATrainConfig.from_dict(train_config)
        if base_model:
            cfg.base_model = base_model
        self.train_config = cfg
        model = base_model or (None if adapter and Path(adapter).exists() else cfg.base_model)
        kwargs.setdefault("prompt_style", "sft")
        kwargs.setdefault("load_in_4bit", cfg.load_in_4bit)
        super().__init__(model=model, adapter=adapter if adapter and Path(adapter).exists() else None, **kwargs)
        self.output_dir = adapter
        self.name = "lora" + ("+cd" if self.constrained else "")

    def fit(self, train: Sequence["BenchExample"], dev: Sequence["BenchExample"] | None = None) -> "LoRAGrounder":
        if not self.output_dir:
            raise ValueError("LoRAGrounder.fit needs an output (adapter) directory")
        out = train_lora(train, self.output_dir, self.train_config, dev, model=self._model, tokenizer=self._tok,
                         return_model=True)
        self.adapter, self.model_name = self.output_dir, self.train_config.base_model
        self._model, self._tok = out["model"], out["tokenizer"]
        self._tok.padding_side = "left"
        return self
