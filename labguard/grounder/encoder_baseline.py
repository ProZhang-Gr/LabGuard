"""Fine-tuned encoder baseline ("SciBERT", Table 2).

An encoder cannot generate free-form predicates, so the baseline treats
grounding as multi-task classification over the [CLS] representation:

  type, severity, intervention   softmax heads
  task tags, substances          multi-label (sigmoid) heads over TASK_TAGS / registry ids
  predicate                      softmax over predicate *templates* seen in training
                                 (numbers abstracted to slots), whose slots are then
                                 filled with the unit-normalised numbers of the rule

Unseen predicate shapes cannot be produced, which is why this baseline has the
lowest predicate scores in the paper.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from labguard.grounder.base import EMPTY_IR, Grounder
from labguard.grounder.regex_grounder import extract_quantities, fmt_num
from labguard.ir.registry import Registry
from labguard.ir.schema import LabGuardIR, sort_tags
from labguard.vocab import CONSTRAINT_TYPES, INTERVENTIONS, SEVERITIES, TASK_TAGS

if TYPE_CHECKING:
    from labguard.bench.example import BenchExample

_PRED_NUM = re.compile(r"(?<![\w.'])(\d+(?:\.\d+)?)(?![\w'])")


def predicate_template(pred: str) -> str:
    return _PRED_NUM.sub("<NUM>", pred)


def fill_template(template: str, text: str, defaults: Sequence[str] = ()) -> str:
    nums = [fmt_num(q.value) for q in extract_quantities(text)] or list(defaults)
    it = iter(nums)
    return re.sub("<NUM>", lambda _: next(it, "0"), template)


@dataclass
class EncoderTrainConfig:
    model: str = "allenai/scibert_scivocab_uncased"
    lr: float = 3e-5
    head_lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 16
    max_len: int = 128
    dropout: float = 0.1
    seed: int = 0


class EncoderGrounder(Grounder):
    name = "scibert"
    trainable = True

    def __init__(self, model: str | None = None, checkpoint: str | None = None,
                 config: EncoderTrainConfig | None = None, registry: Registry | None = None,
                 device: str | None = None) -> None:
        super().__init__(registry)
        self.cfg = config or EncoderTrainConfig()
        if model:
            self.cfg.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.subs_vocab = list(self.registry.ids)
        self.templates: list[str] = []
        self.template_defaults: dict[str, list[str]] = {}
        self._net: Any = None
        self._tok: Any = None
        if checkpoint and (Path(checkpoint) / "labguard_encoder.json").exists():
            self.load(checkpoint)

    # -- network ---------------------------------------------------------------------
    def _build(self) -> None:
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        cfg, n_templates, n_subs = self.cfg, max(1, len(self.templates)), len(self.subs_vocab)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = AutoModel.from_pretrained(cfg.model)
                h = self.encoder.config.hidden_size
                self.drop = nn.Dropout(cfg.dropout)
                self.heads = nn.ModuleDict({
                    "type": nn.Linear(h, len(CONSTRAINT_TYPES)), "severity": nn.Linear(h, len(SEVERITIES)),
                    "intervention": nn.Linear(h, len(INTERVENTIONS)), "tags": nn.Linear(h, len(TASK_TAGS)),
                    "substances": nn.Linear(h, n_subs), "predicate": nn.Linear(h, n_templates)})

            def forward(self, **enc: Any) -> dict[str, Any]:
                cls = self.drop(self.encoder(**enc).last_hidden_state[:, 0])
                return {k: head(cls) for k, head in self.heads.items()}

        self._tok = AutoTokenizer.from_pretrained(cfg.model)
        self._net = Net().to(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _targets(self, examples: Sequence["BenchExample"]) -> dict[str, Any]:
        import torch

        def multi_hot(items: Sequence[str], vocab: Sequence[str]) -> list[float]:
            return [float(v in items) for v in vocab]

        tidx = {t: i for i, t in enumerate(self.templates)}
        return {
            "type": torch.tensor([CONSTRAINT_TYPES.index(e.ir.constraint_type) for e in examples]),
            "severity": torch.tensor([SEVERITIES.index(e.ir.severity) for e in examples]),
            "intervention": torch.tensor([INTERVENTIONS.index(e.ir.intervention) for e in examples]),
            "tags": torch.tensor([multi_hot(e.ir.action_scope, TASK_TAGS) for e in examples]),
            "substances": torch.tensor([multi_hot(e.ir.substances, self.subs_vocab) for e in examples]),
            "predicate": torch.tensor([tidx.get(predicate_template(e.ir.predicate), 0) for e in examples]),
        }

    def fit(self, train: Sequence["BenchExample"], dev: Sequence["BenchExample"] | None = None) -> "EncoderGrounder":
        import torch
        import torch.nn.functional as F

        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        counts: dict[str, int] = {}
        for e in train:
            t = predicate_template(e.ir.predicate)
            counts[t] = counts.get(t, 0) + 1
            self.template_defaults.setdefault(t, _PRED_NUM.findall(e.ir.predicate))
        self.templates = sorted(counts, key=lambda t: -counts[t])
        self._build()
        net, tok = self._net, self._tok
        enc_params = list(net.encoder.parameters())
        head_params = list(net.heads.parameters())
        opt = torch.optim.AdamW([{"params": enc_params, "lr": self.cfg.lr},
                                 {"params": head_params, "lr": self.cfg.head_lr}], weight_decay=0.01)
        y = self._targets(train)
        dev_ = next(net.parameters()).device
        net.train()
        for epoch in range(self.cfg.epochs):
            order = list(range(len(train)))
            random.shuffle(order)
            tot = 0.0
            for b in range(0, len(order), self.cfg.batch_size):
                idx = order[b:b + self.cfg.batch_size]
                enc = tok([train[i].text for i in idx], padding=True, truncation=True, max_length=self.cfg.max_len,
                          return_tensors="pt").to(dev_)
                out = net(**enc)
                yy = {k: v[idx].to(dev_) for k, v in y.items()}
                loss = sum(F.cross_entropy(out[k], yy[k]) for k in ("type", "severity", "intervention", "predicate"))
                loss = loss + sum(F.binary_cross_entropy_with_logits(out[k], yy[k]) for k in ("tags", "substances"))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            print(f"[scibert] epoch {epoch} loss {tot / len(train):.4f}")
        net.eval()
        if self.checkpoint:
            self.save(self.checkpoint)
        return self

    def ground_batch(self, texts: Sequence[str]) -> list[LabGuardIR]:
        import torch

        if self._net is None:
            return [EMPTY_IR for _ in texts]
        out_irs = []
        dev_ = next(self._net.parameters()).device
        for b in range(0, len(texts), 32):
            chunk = list(texts[b:b + 32])
            enc = self._tok(chunk, padding=True, truncation=True, max_length=self.cfg.max_len,
                            return_tensors="pt").to(dev_)
            with torch.no_grad():
                out = self._net(**enc)
            for i, text in enumerate(chunk):
                tags = [t for t, p in zip(TASK_TAGS, torch.sigmoid(out["tags"][i]).tolist()) if p > 0.5]
                subs = [s for s, p in zip(self.subs_vocab, torch.sigmoid(out["substances"][i]).tolist()) if p > 0.5]
                tmpl = self.templates[int(out["predicate"][i].argmax())]
                out_irs.append(LabGuardIR(
                    CONSTRAINT_TYPES[int(out["type"][i].argmax())], tuple(self.registry.sort_ids(subs)),
                    fill_template(tmpl, text, self.template_defaults.get(tmpl, ())), sort_tags(tags),
                    SEVERITIES[int(out["severity"][i].argmax())], INTERVENTIONS[int(out["intervention"][i].argmax())]))
        return out_irs

    def ground(self, text: str) -> LabGuardIR:
        return self.ground_batch([text])[0]

    # -- persistence -----------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / "model.pt")
        (path / "labguard_encoder.json").write_text(json.dumps(
            {"config": asdict(self.cfg), "templates": self.templates, "template_defaults": self.template_defaults,
             "substances": self.subs_vocab}, indent=1), encoding="utf-8")

    def load(self, path: str | Path) -> "EncoderGrounder":
        import torch

        path = Path(path)
        meta = json.loads((path / "labguard_encoder.json").read_text(encoding="utf-8"))
        self.cfg = EncoderTrainConfig(**meta["config"])
        self.templates, self.subs_vocab = meta["templates"], meta["substances"]
        self.template_defaults = meta.get("template_defaults", {})
        self._build()
        self._net.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        self._net.eval()
        return self

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.cfg.model, "templates": len(self.templates),
                "epochs": self.cfg.epochs, "lr": self.cfg.lr}
