"""Text serialization of LabGuard-IR used as the grounder's output format.

Canonical target (one field per line, no space after '=' so that the closed
field prefixes tokenize identically in training and constrained decoding):

    type=chemical_incompatibility
    substances=HCl;NaClO
    predicate=not incompatible(src, dst)
    scope=pour;transfer;mix
    severity=high
    intervention=block

The parser is tolerant: it also accepts `key: value` lines, JSON objects
(possibly wrapped in markdown fences or surrounded by chatter) and long key
names (constraint_type / action_scope).
"""

from __future__ import annotations

import json
import re
from typing import Any

from labguard.ir.schema import IR_FIELDS, SHORT_KEYS, LabGuardIR

LINE_KEYS: tuple[str, ...] = tuple(SHORT_KEYS[f] for f in IR_FIELDS)
_KEY_ALIASES: dict[str, str] = {
    "type": "constraint_type", "constraint_type": "constraint_type", "t": "constraint_type",
    "constrainttype": "constraint_type",
    "substances": "substances", "substance": "substances", "s": "substances", "entities": "substances",
    "predicate": "predicate", "p": "predicate", "condition": "predicate",
    "scope": "action_scope", "action_scope": "action_scope", "actionscope": "action_scope",
    "tags": "action_scope", "task_tags": "action_scope",
    "severity": "severity", "v": "severity",
    "intervention": "intervention", "i": "intervention", "mode": "intervention",
}
LIST_SEP = ";"


def ir_to_text(ir: LabGuardIR) -> str:
    return "\n".join([
        f"type={ir.constraint_type}",
        f"substances={LIST_SEP.join(ir.substances)}",
        f"predicate={ir.predicate}",
        f"scope={LIST_SEP.join(ir.action_scope)}",
        f"severity={ir.severity}",
        f"intervention={ir.intervention}",
    ])


def _split_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    value = str(value).strip().strip("[]{}()")
    parts = re.split(r"[;,|]", value)
    return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]


def _from_mapping(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = _KEY_ALIASES.get(re.sub(r"[^a-z_]", "", str(k).lower()))
        if key and key not in out:
            out[key] = v
    return out


def _try_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return _from_mapping(obj) if isinstance(obj, dict) else None


def parse_fields(text: str) -> dict[str, Any]:
    """Parse raw grounder output into a partial field dict (missing keys absent)."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    if "{" in text:
        js = _try_json(text)
        if js:
            return js
    fields: dict[str, Any] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*[-*]?\s*([A-Za-z_ ]+?)\s*(?:=|:)\s*(.*)$", line)
        if not m:
            continue
        key = _KEY_ALIASES.get(re.sub(r"[^a-z_]", "", m.group(1).lower().replace(" ", "_")))
        if key and key not in fields:
            fields[key] = m.group(2).strip()
    return fields


def fields_to_ir(fields: dict[str, Any]) -> LabGuardIR:
    return LabGuardIR(
        constraint_type=str(fields.get("constraint_type", "") or "").strip(),
        substances=tuple(_split_list(fields.get("substances"))),
        predicate=str(fields.get("predicate", "") or "").strip(),
        action_scope=tuple(t.lower() for t in _split_list(fields.get("action_scope"))),
        severity=str(fields.get("severity", "") or "").strip().lower(),
        intervention=str(fields.get("intervention", "") or "").strip().lower(),
    )


def text_to_ir(text: str) -> LabGuardIR:
    """Never raises: missing fields become empty strings / empty tuples."""
    return fields_to_ir(parse_fields(text))
