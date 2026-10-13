"""Prompted LLM grounders over an OpenAI-compatible chat API (Table 2 "0-shot" rows).

Works with any OpenAI-compatible endpoint, e.g. a local vLLM server

    vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000
    OPENAI_BASE_URL=http://localhost:8000/v1 python scripts/eval_grounding.py --grounder llm:Qwen/Qwen2.5-72B-Instruct

or a hosted API (DeepSeek-V3.1: base_url https://api.deepseek.com, model deepseek-chat).
`shots > 0` turns the zero-shot prompt into a few-shot prompt with examples
retrieved from the training split (nearest neighbours by TF-IDF).
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Sequence

from labguard.grounder.base import Grounder, clean_ir
from labguard.grounder.prompts import chat_messages
from labguard.ir.registry import Registry
from labguard.ir.schema import LabGuardIR
from labguard.ir.serialize import text_to_ir

if TYPE_CHECKING:
    from labguard.bench.example import BenchExample

_THINK = re.compile(r"<think>.*?</think>", re.S)


class ChatLLMGrounder(Grounder):
    name = "llm"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, shots: int = 0,
                 temperature: float = 0.0, max_tokens: int = 256, retries: int = 3, workers: int = 4,
                 registry: Registry | None = None, client: Any = None) -> None:
        super().__init__(registry)
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.shots = shots
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.workers = workers
        self._client = client
        self._pool: list["BenchExample"] = []
        self._index: Any = None
        self.trainable = shots > 0
        self.name = f"llm:{model}" + (f"@{shots}shot" if shots else "")

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def fit(self, train: Sequence["BenchExample"], dev: Sequence["BenchExample"] | None = None) -> "ChatLLMGrounder":
        """Few-shot pool = training split (zero-shot grounders ignore it)."""
        if self.shots:
            from labguard.grounder.retrieval import TfidfIndex

            self._pool = list(train)
            self._index = TfidfIndex([e.text for e in self._pool])
        return self

    def select_shots(self, text: str) -> list["BenchExample"]:
        if not self.shots or self._index is None:
            return []
        return [self._pool[i] for i in self._index.nearest(text, self.shots)][::-1]

    def complete(self, messages: list[dict[str, str]]) -> str:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                        temperature=self.temperature, max_tokens=self.max_tokens)
                return _THINK.sub("", r.choices[0].message.content or "").strip()
            except Exception as exc:  # rate limits / transient network errors
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last}")

    def raw(self, text: str) -> str:
        return self.complete(chat_messages(text, self.select_shots(text), self.registry))

    def ground_batch(self, texts: Sequence[str]) -> list[LabGuardIR]:
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
            raw = list(ex.map(self.raw, texts))
        self.last_raw = raw
        return [clean_ir(text_to_ir(r), self.registry) for r in raw]

    def ground(self, text: str) -> LabGuardIR:
        return self.ground_batch([text])[0]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "base_url": self.base_url, "shots": self.shots}
