"""Schema-constrained decoding for the LoRA grounder ("+ Constraint decoding", Table 2).

`IRGrammar` is a character-level prefix checker for the six-line IR
serialization: field keys must appear in order, closed fields (type, severity,
intervention) and list fields (substances, scope) may only spell registry /
vocabulary values, and the predicate line may only use identifiers,
attributes and quoted literals of the predicate language; a predicate line
can only be terminated when it parses and passes the vocabulary check.

`IRLogitsProcessor` applies the checker during Hugging Face generation: at
each step it keeps the highest-scoring candidate tokens whose text extends
the output to a valid prefix and masks everything else.  This guarantees
Schema Valid output (up to the max_new_tokens budget) without retraining.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from labguard.ir.predicate import FUNCTIONS, ROOTS, HAZARD_CLASSES, Predicate
from labguard.ir.registry import Registry, default_registry
from labguard.ir.serialize import LINE_KEYS, LIST_SEP
from labguard.vocab import CONSTRAINT_TYPES, INTERVENTIONS, SEVERITIES, TASK_TAGS

_PRED_CHARS = re.compile(r"^[A-Za-z0-9_ .(),'<>=!+\-*/]*$")
_KEYWORDS = frozenset({"and", "or", "not", "in", "True", "False"})
_CMP_CHARS = frozenset("<>=!")
_CMP_OPS = ("<", "<=", ">", ">=", "==", "!=")
_PREV_OP = re.compile(r"(?:[<>=!+*/(,]|^)\s*$")


@dataclass(frozen=True)
class PrefixState:
    ok: bool  # text is a prefix of some valid serialization
    complete: bool  # text is itself a complete valid serialization
    forced: str = ""  # deterministic continuation, if any (used when no candidate token fits)


class IRGrammar:
    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry = registry or default_registry()
        self.closed: dict[str, tuple[str, ...]] = {
            "type": CONSTRAINT_TYPES, "severity": SEVERITIES, "intervention": INTERVENTIONS}
        self.lists: dict[str, tuple[str, ...]] = {"substances": tuple(registry.ids), "scope": TASK_TAGS}
        self.idents = frozenset(ROOTS) | frozenset(FUNCTIONS) | _KEYWORDS
        flags = {f for fs in registry.equipment_flags.values() for f in fs}
        self.literals = frozenset(registry.ids) | frozenset(registry.zones) | frozenset(registry.ppe_items) \
            | frozenset(registry.waste_streams) | frozenset(flags) | frozenset(registry.step_names()) \
            | frozenset(HAZARD_CLASSES)
        self._literal_prefixes = {lit[:k] for lit in self.literals for k in range(len(lit) + 1)}
        self._ident_prefixes = {w[:k] for w in self.idents for k in range(len(w) + 1)}
        self._attr_prefixes = {root: {a[:k] for a in attrs for k in range(len(a) + 1)} for root, attrs in ROOTS.items()}

    # -- field values ------------------------------------------------------------
    def _closed(self, key: str, value: str, final: bool) -> bool:
        opts = self.closed[key]
        return value in opts if final else any(o.startswith(value) for o in opts)

    def _list(self, key: str, value: str, final: bool) -> bool:
        if value == "":
            return True
        items = value.split(LIST_SEP)
        opts = self.lists[key]
        done, last = items[:-1], items[-1]
        if any(i not in opts for i in done) or len(set(done)) != len(done):
            return False
        if final:
            return last in opts and last not in done
        return any(o.startswith(last) for o in opts if o not in done)

    def predicate_prefix_ok(self, text: str) -> bool:
        if not _PRED_CHARS.match(text):
            return False
        depth = 0
        i, n = 0, len(text)
        prev_root: str | None = None
        while i < n:
            ch = text[i]
            if ch == "'":
                j = text.find("'", i + 1)
                lit = text[i + 1:] if j < 0 else text[i + 1:j]
                if j < 0:
                    return lit in self._literal_prefixes
                if lit not in self.literals:
                    return False
                i = j + 1
                prev_root = None
                continue
            if ch.isalpha() or ch == "_":
                j = i
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                word = text[i:j]
                at_end = j == n
                if prev_root is not None:  # attribute after "root."
                    if not (word in self._attr_prefixes[prev_root] if at_end else word in ROOTS[prev_root]):
                        return False
                    prev_root = None
                else:
                    if not (word in self._ident_prefixes if at_end else word in self.idents):
                        return False
                    if j < n and text[j] == ".":
                        if word not in ROOTS:
                            return False
                        prev_root = word
                        j += 1
                        if j == n:
                            return True
                        if not (text[j].isalpha() or text[j] == "_"):
                            return False
                i = j
                continue
            if ch.isdigit():
                j = i
                while j < n and (text[j].isdigit() or text[j] == "."):
                    j += 1
                whole, _, frac = text[i:j].partition(".")
                if "." in frac or len(whole) > 6 or len(frac) > 4:  # IR constants are small, short numbers
                    return False
                i = j
                prev_root = None
                continue
            if ch in _CMP_CHARS:
                j = i
                while j < n and text[j] in _CMP_CHARS:
                    j += 1
                run = text[i:j]
                if _PREV_OP.search(text[:i]):  # operator with no left operand ("> <", leading "==")
                    return False
                if j == n:
                    if not any(op.startswith(run) for op in _CMP_OPS):
                        return False
                elif run not in _CMP_OPS:
                    return False
                i = j
                prev_root = None
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
            elif ch == ".":
                return False
            prev_root = None
            i += 1
        return True

    def predicate_complete(self, text: str) -> bool:
        if not text.strip() or text != text.strip():
            return False
        p = Predicate.try_parse(text, self.registry)
        return p is not None and not p.vocabulary_errors()

    def _value_ok(self, key: str, value: str, final: bool) -> bool:
        if key in self.closed:
            return self._closed(key, value, final)
        if key in self.lists:
            return self._list(key, value, final)
        return self.predicate_complete(value) if final else self.predicate_prefix_ok(value)

    # -- whole output ----------------------------------------------------------------
    def check(self, text: str) -> PrefixState:
        lines = text.split("\n")
        if len(lines) > len(LINE_KEYS) + 1 or (len(lines) == len(LINE_KEYS) + 1 and lines[-1] != ""):
            return PrefixState(False, False)
        for k, line in enumerate(lines):
            key = LINE_KEYS[k] if k < len(LINE_KEYS) else None
            final = k < len(lines) - 1
            if key is None:  # trailing newline after the last field
                return PrefixState(True, True)
            head = key + "="
            if not line.startswith(head):
                if final or not head.startswith(line):
                    return PrefixState(False, False)
                return PrefixState(True, False, head[len(line):])
            value = line[len(head):]
            if not self._value_ok(key, value, final):
                return PrefixState(False, False)
            if not final:
                complete_here = self._value_ok(key, value, True)
                is_last = k == len(LINE_KEYS) - 1
                forced = self._forced_value(key, value)
                if complete_here and not forced:
                    forced = "" if is_last else "\n"
                return PrefixState(True, complete_here and is_last, forced)
        return PrefixState(True, False)

    def _forced_value(self, key: str, value: str) -> str:
        """Unique completion of a closed value prefix (e.g. "chemical_inc" -> "ompatibility")."""
        if key not in self.closed:
            return ""
        opts = [o for o in self.closed[key] if o.startswith(value)]
        return opts[0][len(value):] if len(opts) == 1 else ""

    def is_valid(self, text: str) -> bool:
        return self.check(text.rstrip("\n") if text.endswith("\n") else text).complete


class IRLogitsProcessor:
    """Hugging Face `LogitsProcessor` enforcing `IRGrammar` on the generated suffix.

    `prompt_len` is the (padded) prompt length; generation must use left
    padding so that every row's continuation starts at the same position.
    """

    def __init__(self, tokenizer: Any, prompt_len: int, grammar: IRGrammar | None = None, top_k: int = 64,
                 wide_k: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.grammar = grammar or IRGrammar()
        self.top_k = top_k
        self.wide_k = wide_k
        eos = tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        self._tok_text: dict[int, str] = {}

    def token_text(self, tid: int) -> str:
        s = self._tok_text.get(tid)
        if s is None:
            s = self.tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            self._tok_text[tid] = s
        return s

    def allowed(self, generated: str, candidates: Sequence[int]) -> list[int]:
        state = self.grammar.check(generated)
        out = []
        for tid in candidates:
            if tid in self.eos_ids:
                if state.complete:
                    out.append(tid)
                continue
            piece = self.token_text(tid)
            if piece and "�" not in piece and self.grammar.check(generated + piece).ok:
                out.append(tid)
        return out

    def forced_token(self, generated: str) -> int | None:
        state = self.grammar.check(generated)
        if state.complete:
            return next(iter(self.eos_ids), None)
        if state.forced:
            ids = self.tokenizer.encode(state.forced, add_special_tokens=False)
            return ids[0] if ids else None
        return None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        out = torch.full_like(scores, float("-inf"))
        for row in range(input_ids.shape[0]):
            gen = self.tokenizer.decode(input_ids[row, self.prompt_len:], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            keep: list[int] = []
            for k in (self.top_k, self.wide_k):
                cand = torch.topk(scores[row], min(k, scores.shape[-1])).indices.tolist()
                keep = self.allowed(gen, cand)
                if keep:
                    break
            if not keep:
                forced = self.forced_token(gen)
                keep = [forced] if forced is not None else [int(scores[row].argmax())]
            idx = torch.tensor(keep, device=scores.device)
            out[row, idx] = scores[row, idx]
        return out
