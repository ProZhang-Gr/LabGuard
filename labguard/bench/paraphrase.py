"""Paraphrase augmentation with semantic-equivalence review (Sec. 3.3, Appendix D.1).

Each seed rule receives three paraphrases.  A paraphrase inherits the seed
label only if it passes `review_paraphrase`, which rejects candidates that
add or drop a registry mention, change a numeric value, or change the
polarity / strength of the obligation.

`TemplateParaphraser` is offline and deterministic given a seed: it draws
from the alternative wordings produced by the seed generator and applies
meaning-preserving rewrites (severity-cue swaps within the same strength
class, lexical substitutions, alias swaps).  `LLMParaphraser` calls an
OpenAI-compatible chat endpoint; its outputs go through the same review.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from labguard.bench.templates import NEG, POS, RNEG, RPOS, tidy
from labguard.ir.registry import Registry, default_registry

_NUM = re.compile(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?")
_NEGATION = re.compile(r"\b(never|not|no|none|nothing|prohibited|forbidden|reject|block|abort|n't)\b|n't\b",
                       re.IGNORECASE)
_STRENGTH = {
    "high": re.compile(r"\b(never|always|must|shall|mandatory|strictly|under no circumstances|prohibited|"
                       r"forbidden|only)\b", re.IGNORECASE),
    "low": re.compile(r"\b(where possible|preferabl[ey]|good practice|try not)\b", re.IGNORECASE),
}

# (pattern, replacements) -- meaning-preserving lexical substitutions
LEXICAL_SWAPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"\bmake sure that\b", ("ensure that", "verify that")),
    (r"\bcheck that\b", ("verify that", "confirm that", "make sure that")),
    (r"\bverify that\b", ("check that", "confirm that")),
    (r"\bconfirm that\b", ("check that", "verify that")),
    (r"\bDo not\b", ("Don't",)),
    (r"\bdo not\b", ("don't",)),
    (r"\bat or below\b", ("no higher than", "not above")),
    (r"\bat most\b", ("no more than", "a maximum of")),
    (r"\bno more than\b", ("at most",)),
    (r"\bexceed\b", ("go above", "go beyond")),
    (r"\bprohibited\b", ("forbidden", "not permitted")),
    (r"\bforbidden\b", ("prohibited",)),
    (r"\bswitched on\b", ("turned on",)),
    (r"\bswitched off\b", ("turned off",)),
    (r"\bthe robot\b", ("the robot arm", "the manipulator")),
    (r"\bThe robot\b", ("The robot arm", "The manipulator")),
    (r"\bthe controller\b", ("the robot controller", "the motion controller")),
    (r"\bThe controller\b", ("The robot controller", "The motion controller")),
    (r"\bcarry\b", ("transport",)),
    (r"\bcarrying\b", ("transporting",)),
    (r"\bliquids\b", ("solutions",)),
    (r"\bvessel\b", ("receptacle",)),
    (r"(?<!waste )\bcontainers\b", ("vessels",)),
    (r"(?<!waste )\bcontainer\b", ("vessel",)),
    (r"\bslowly\b", ("gently", "carefully")),
    (r"\bStore\b", ("Keep",)),
    (r"\bmust be worn\b", ("is required",)),
    (r"\bWear\b", ("Put on",)),
    (r"\bwear\b", ("put on",)),
    (r"\bgrip force\b", ("gripping force",)),
    (r"\bspeed\b", ("velocity",)),
    (r"\bend effector\b", ("gripper",)),
    (r"\bthe arm\b", ("the robot arm",)),
)
WRAPPERS: tuple[str, ...] = ("Safety rule: {r}", "Lab policy: {r}", "Note: {r}", "Important: {r}",
                             "Standard operating procedure: {r}")


def _numbers(text: str) -> list[float]:
    return sorted(float(x) for x in _NUM.findall(text))


def _mention_ids(text: str, registry: Registry, chemicals: bool = True) -> set[str]:
    return {sid for sid, _ in registry.find_mentions(text) if registry.is_chemical(sid) == chemicals}


def _strength(text: str) -> str:
    if _STRENGTH["low"].search(text):
        return "low"
    return "high" if _STRENGTH["high"].search(text) else "medium"


@dataclass
class Review:
    ok: bool
    reasons: list[str]


def _negated(text: str) -> bool:
    return bool(_NEGATION.search(text))


def review_paraphrase(seed_text: str, candidate: str, registry: Registry | None = None,
                      severity: str | None = None, references: Sequence[str] = ()) -> Review:
    """Semantic-equivalence review used before a paraphrase inherits the seed label.

    Chemical mentions and numeric values must match the seed exactly; equipment
    mentions and the polarity of the wording ("never X" vs "always Y") must
    match the seed or one of the reviewed reference wordings of the same rule.
    """
    registry = registry or default_registry()
    refs = [seed_text, *references]
    reasons = []
    if candidate.strip().lower() == seed_text.strip().lower():
        reasons.append("identical to seed")
    if _numbers(candidate) != _numbers(seed_text):
        reasons.append("numbers changed")
    if _mention_ids(candidate, registry) != _mention_ids(seed_text, registry):
        reasons.append("chemical mentions changed")
    allowed = set().union(*(_mention_ids(r, registry, chemicals=False) for r in refs))
    if not _mention_ids(candidate, registry, chemicals=False) <= allowed:
        reasons.append("new equipment mention")
    if _negated(candidate) not in {_negated(r) for r in refs}:
        reasons.append("polarity changed")
    if _strength(candidate) not in {_strength(r) for r in refs}:
        reasons.append("obligation strength changed")
    if not 3 <= len(candidate.split()) <= 3 * len(seed_text.split()) + 10:
        reasons.append("length out of range")
    return Review(not reasons, reasons)


class Paraphraser:
    method = "base"

    def candidates(self, text: str, alt_texts: Sequence[str], severity: str,
                   rng: np.random.Generator) -> list[tuple[str, str]]:
        """Returns (candidate text, method) pairs in preference order."""
        raise NotImplementedError

    def paraphrase(self, text: str, alt_texts: Sequence[str], severity: str, n: int,
                   rng: np.random.Generator, registry: Registry | None = None) -> list[tuple[str, str]]:
        registry = registry or default_registry()
        out: list[tuple[str, str]] = []
        seen = {text.strip().lower()}
        for cand, method in self.candidates(text, alt_texts, severity, rng):
            key = cand.strip().lower()
            if key in seen:
                continue
            if review_paraphrase(text, cand, registry, severity, alt_texts).ok:
                out.append((cand, method))
                seen.add(key)
            if len(out) == n:
                break
        return out


class TemplateParaphraser(Paraphraser):
    method = "template"

    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or default_registry()

    # -- rewrite operators -----------------------------------------------------
    def swap_cue(self, text: str, severity: str, rng: np.random.Generator) -> str | None:
        for table in (RNEG, RPOS, NEG, POS):
            cues = sorted(table[severity], key=len, reverse=True)
            for cue in cues:
                if text.startswith(cue + " ") or f". {cue} " in text or f"; {cue} " in text or f": {cue} " in text:
                    others = [c for c in table[severity] if c != cue]
                    if not others:
                        return None
                    repl = others[int(rng.integers(len(others)))]
                    return tidy(re.sub(re.escape(cue) + r"\s", repl + " ", text, count=1))
        return None

    def swap_lexical(self, text: str, rng: np.random.Generator, k: int = 2) -> str | None:
        options = [(pat, reps) for pat, reps in LEXICAL_SWAPS if re.search(pat, text)]
        if not options:
            return None
        idx = rng.permutation(len(options))[:k]
        out = text
        for i in idx:
            pat, reps = options[int(i)]
            out = re.sub(pat, reps[int(rng.integers(len(reps)))], out, count=1)
        return tidy(out) if out != text else None

    def swap_alias(self, text: str, rng: np.random.Generator) -> str | None:
        mentions = [(sid, span) for sid, span in self.registry.find_mentions(text)
                    if self.registry.is_chemical(sid)]
        if not mentions:
            return None
        sid, (a, b) = mentions[int(rng.integers(len(mentions)))]
        current = text[a:b]
        names = [n for n in self.registry.surface_names(sid) if n.lower() != current.lower() and "_" not in n]
        if not names:
            return None
        new = names[int(rng.integers(len(names)))]
        if a == 0 or text[max(0, a - 2):a] in (". ", "! ", "? "):
            new = new[0].upper() + new[1:] if new[:1].islower() else new
        out = text[:a] + new + text[b:]
        # every other occurrence of the same surface form follows the swap
        out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(current) + r"(?![A-Za-z0-9])", new, out)
        return tidy(out)

    def wrap(self, text: str, rng: np.random.Generator) -> str:
        w = WRAPPERS[int(rng.integers(len(WRAPPERS)))]
        body = text[0].lower() + text[1:] if text[:2].isalpha() and not text[:2].isupper() else text
        return tidy(w.format(r=body))

    def candidates(self, text: str, alt_texts: Sequence[str], severity: str,
                   rng: np.random.Generator) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = [(t, "template") for t in alt_texts]
        bases = [text, *alt_texts]
        ops: list[tuple[str, Callable[[str], str | None]]] = [
            ("cue", lambda t: self.swap_cue(t, severity, rng)),
            ("lexical", lambda t: self.swap_lexical(t, rng)),
            ("alias", lambda t: self.swap_alias(t, rng)),
        ]
        for _ in range(3):
            for base in bases:
                for name, op in [ops[int(i)] for i in rng.permutation(len(ops))]:
                    c = op(base)
                    if c:
                        out.append((c, name))
                        c2 = ops[int(rng.integers(len(ops)))][1](c)
                        if c2:
                            out.append((c2, f"{name}+"))
        # interleave: alternatives first, then rewrites, shuffled within each group
        alts, rewrites = out[:len(alt_texts)], out[len(alt_texts):]
        rewrites = [rewrites[int(i)] for i in rng.permutation(len(rewrites))]
        tail = [(self.wrap(b, rng), "wrapper") for b in bases for _ in range(2)]
        return alts + rewrites + tail


class LLMParaphraser(Paraphraser):
    """OpenAI-compatible chat-completions paraphraser (e.g. GPT-4o, a vLLM server)."""

    method = "llm"
    PROMPT = ("Paraphrase the following laboratory safety rule {n} different ways. Keep every chemical name, "
              "number, unit and the strength of the obligation (must / should / where possible) unchanged, and "
              "do not add or remove any condition. Return a JSON list of strings only.\n\nRule: {rule}")

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.9, n: int = 6, fallback: Paraphraser | None = None) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.temperature = temperature
        self.n = n
        self.fallback = fallback or TemplateParaphraser()
        self._client: Any = None

    def _chat(self, prompt: str) -> str:
        if self._client is None:
            from openai import OpenAI  # optional dependency

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        r = self._client.chat.completions.create(model=self.model, temperature=self.temperature,
                                                 messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""

    @staticmethod
    def parse(reply: str) -> list[str]:
        m = re.search(r"\[.*\]", reply, re.DOTALL)
        if m:
            try:
                return [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip() for line in reply.splitlines() if line.strip()]

    def candidates(self, text: str, alt_texts: Sequence[str], severity: str,
                   rng: np.random.Generator) -> list[tuple[str, str]]:
        try:
            outs = [(tidy(t), "llm") for t in self.parse(self._chat(self.PROMPT.format(n=self.n, rule=text)))]
        except Exception:  # network / package errors -> offline fallback
            outs = []
        return outs + self.fallback.candidates(text, alt_texts, severity, rng)


def build_paraphraser(spec: str | None = None, **kwargs: Any) -> Paraphraser:
    if spec in (None, "", "template"):
        return TemplateParaphraser(kwargs.get("registry"))
    if spec.startswith("llm"):
        model = spec.split(":", 1)[1] if ":" in spec else kwargs.pop("model", "gpt-4o-mini")
        return LLMParaphraser(model=model, **{k: v for k, v in kwargs.items() if k != "registry"})
    raise ValueError(f"unknown paraphraser {spec!r}")


def paraphrase_all(texts: Iterable[tuple[str, Sequence[str], str]], paraphraser: Paraphraser, n: int,
                   rng: np.random.Generator, registry: Registry | None = None) -> list[list[tuple[str, str]]]:
    return [paraphraser.paraphrase(t, alts, sev, n, rng, registry) for t, alts, sev in texts]
