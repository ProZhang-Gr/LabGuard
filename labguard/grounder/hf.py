"""Local Hugging Face causal-LM grounders (Base Grounder / LoRA / + constraint decoding).

  HFGrounder("Qwen/Qwen3-8B", prompt_style="chat")            Base Grounder (Qwen3-8B), zero-shot
  HFGrounder("Qwen/Qwen3-8B", adapter="outputs/lora")         + LoRA fine-tuning
  HFGrounder(..., adapter="outputs/lora", constrained=True)   + Constraint decoding

torch / transformers / peft are imported lazily so that the rest of the
package works without them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from labguard.grounder.base import Grounder, clean_ir
from labguard.grounder.constrained import IRGrammar, IRLogitsProcessor
from labguard.grounder.prompts import chat_messages, sft_prompt
from labguard.ir.registry import Registry
from labguard.ir.schema import LabGuardIR
from labguard.ir.serialize import text_to_ir

_THINK = re.compile(r"<think>.*?</think>", re.S)


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(name: str | None) -> Any:
    import torch

    if name in (None, "auto"):
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32}[name]


def load_tokenizer(name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(name_or_path: str, load_in_4bit: bool = False, dtype: str | None = "auto",
                   device: str | None = None) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": resolve_dtype(dtype)}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig  # requires bitsandbytes + CUDA

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    if not load_in_4bit:
        model.to(device or default_device())
    return model


def adapter_base_model(adapter: str | Path) -> str | None:
    """Base-model id recorded in a PEFT adapter directory."""
    for fn in ("labguard_lora.json", "adapter_config.json"):
        p = Path(adapter) / fn
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            base = d.get("base_model") or d.get("base_model_name_or_path")
            if base:
                return str(base)
    return None


class HFGrounder(Grounder):
    """Greedy generation of the six-line IR with an (optionally LoRA-adapted) causal LM."""

    name = "hf"

    def __init__(self, model: str | None = None, adapter: str | None = None, prompt_style: str | None = None,
                 constrained: bool = False, max_new_tokens: int = 160, batch_size: int = 8,
                 load_in_4bit: bool = False, dtype: str | None = "auto", device: str | None = None,
                 shots: Sequence[Any] = (), registry: Registry | None = None,
                 model_obj: Any = None, tokenizer_obj: Any = None) -> None:
        super().__init__(registry)
        if model is None and adapter is not None:
            model = adapter_base_model(adapter)
        if model is None and model_obj is None:
            raise ValueError("HFGrounder needs a base model id/path (or an adapter that records one)")
        self.model_name = model
        self.adapter = adapter
        # fine-tuned adapters were trained on the plain SFT prompt; untuned models get the chat prompt
        self.prompt_style = prompt_style or ("sft" if adapter or model_obj is not None else "chat")
        self.constrained = constrained
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.load_in_4bit = load_in_4bit
        self.dtype = dtype
        self.device = device
        self.shots = list(shots)
        self.grammar = IRGrammar(self.registry) if constrained else None
        self._model = model_obj
        self._tok = tokenizer_obj
        self.name = "hf" + ("+lora" if adapter else "") + ("+cd" if constrained else "")

    # -- loading -----------------------------------------------------------------
    def load(self) -> "HFGrounder":
        if self._tok is None:
            self._tok = load_tokenizer(self.adapter if self.adapter and (Path(self.adapter) / "tokenizer_config.json")
                                       .exists() else self.model_name)
        if self._model is None:
            model = load_causal_lm(self.model_name, self.load_in_4bit, self.dtype, self.device)
            if self.adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter)
            model.eval()
            self._model = model
        self._tok.padding_side = "left"
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        return self

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tok is None:
            self.load()
        return self._tok

    # -- prompting ---------------------------------------------------------------
    def prompt(self, rule: str) -> str:
        if self.prompt_style == "sft":
            return sft_prompt(rule)
        msgs = chat_messages(rule, self.shots, self.registry)
        try:
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        except Exception:  # tokenizer without a chat template
            return "\n\n".join(m["content"] for m in msgs) + "\n"

    def generate(self, texts: Sequence[str]) -> list[str]:
        import torch
        from transformers import LogitsProcessorList

        tok, model = self.tokenizer, self.model
        outs: list[str] = []
        for i in range(0, len(texts), self.batch_size):
            prompts = [self.prompt(t) for t in texts[i:i + self.batch_size]]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            plen = enc["input_ids"].shape[1]
            kwargs: dict[str, Any] = dict(max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=tok.pad_token_id)
            if self.constrained:
                kwargs["logits_processor"] = LogitsProcessorList([IRLogitsProcessor(tok, plen, self.grammar)])
            with torch.no_grad():
                gen = model.generate(**enc, **kwargs)
            for row in gen[:, plen:]:
                outs.append(_THINK.sub("", tok.decode(row, skip_special_tokens=True)).strip())
        return outs

    def ground_batch(self, texts: Sequence[str]) -> list[LabGuardIR]:
        raw = self.generate(list(texts))
        self.last_raw = raw
        return [clean_ir(text_to_ir(r), self.registry) for r in raw]

    def ground(self, text: str) -> LabGuardIR:
        return self.ground_batch([text])[0]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model_name, "adapter": self.adapter,
                "prompt_style": self.prompt_style, "constrained": self.constrained}
