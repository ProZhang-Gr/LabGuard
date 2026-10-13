"""Prompts and the supervised target format of LabGuard-Grounder.

The grounder generates the six-line serialization of `labguard.ir.serialize`:

    type=chemical_incompatibility
    substances=HCl;NaClO
    predicate=not incompatible(src, dst)
    scope=pour;transfer;mix
    severity=high
    intervention=block

`SYSTEM_PROMPT` lists the closed vocabularies so that zero-shot LLMs (Table 2,
"0-shot" rows) and the untuned base grounder see the same schema as the
fine-tuned model.  The fine-tuned model is trained with the short
`INSTRUCTION` prompt (the schema is learned from data, Eq. 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from labguard.ir.predicate import ACTION_ATTRS, CONTAINER_ATTRS, ENV_ATTRS, HAZARD_CLASSES, ROBOT_ATTRS
from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import LabGuardIR
from labguard.ir.serialize import ir_to_text
from labguard.vocab import CONSTRAINT_TYPES, INTERVENTIONS, SEVERITIES, TASK_TAGS

if TYPE_CHECKING:
    from labguard.bench.example import BenchExample

INSTRUCTION = "Translate the laboratory safety rule into LabGuard-IR.\nRule: {rule}\nIR:\n"


def schema_description(registry: Registry | None = None) -> str:
    registry = registry or default_registry()
    flags = "; ".join(f"{e}: {', '.join(fs)}" for e, fs in registry.equipment_flags.items())
    return "\n".join([
        "LabGuard-IR fields (one per line, `key=value`, lists separated by `;`):",
        f"type: one of {', '.join(CONSTRAINT_TYPES)}",
        f"substances: registry ids of the chemicals/equipment the rule is about, from {', '.join(registry.ids)}",
        "predicate: the condition that must HOLD for an action to be safe (the monitor fires when it is false), "
        "written in this expression language:",
        "  containers src (source) and dst (destination) with attributes " + ", ".join(sorted(CONTAINER_ATTRS)),
        "  action." + ", action.".join(sorted(ACTION_ATTRS)),
        "  robot." + ", robot.".join(sorted(ROBOT_ATTRS)) + "; env." + ", env.".join(sorted(ENV_ATTRS)),
        "  functions: incompatible(src, dst), contains(c, 'id'), hazard(c, 'class'), has_ppe('item'), "
        "done('step'), in_zone('zone'), status('equipment', 'flag'), distance('entity'), abs, min, max",
        "  operators: and, or, not, <, <=, >, >=, ==, !=",
        f"  hazard classes: {', '.join(HAZARD_CLASSES)}",
        f"  zones: {', '.join(registry.zones)}",
        f"  PPE items: {', '.join(registry.ppe_items)}",
        f"  waste labels (dst.label): {', '.join(registry.waste_streams)}",
        f"  equipment flags: {flags}",
        f"  steps for done(): {', '.join(sorted(registry.step_names()))}",
        "  units: temperature degC, speed m/s, force N, volume mL, pour_rate mL/s, time s, distance m, "
        "fill ratios as fractions (0.8 = 80%)",
        f"scope: task tags the rule applies to, from {', '.join(TASK_TAGS)}",
        f"severity: one of {', '.join(SEVERITIES)}",
        f"intervention: one of {', '.join(INTERVENTIONS)} (log = record only, modify = clamp the action, "
        "block = stop it)",
    ])


SYSTEM_PROMPT_HEAD = ("You convert natural-language laboratory safety rules for a robot into LabGuard-IR, an "
                      "executable specification that is compiled into a runtime monitor.")
OUTPUT_FORMAT = ("Answer with exactly six lines and nothing else:\n"
                 "type=...\nsubstances=...\npredicate=...\nscope=...\nseverity=...\nintervention=...")


def system_prompt(registry: Registry | None = None) -> str:
    return f"{SYSTEM_PROMPT_HEAD}\n\n{schema_description(registry)}\n\n{OUTPUT_FORMAT}"


def sft_prompt(rule: str) -> str:
    """Prompt part of a supervised example (masked out of the loss)."""
    return INSTRUCTION.format(rule=rule.strip())


def sft_target(ir: LabGuardIR) -> str:
    return ir_to_text(ir)


def chat_messages(rule: str, shots: Sequence["BenchExample"] | Sequence[tuple[str, LabGuardIR]] = (),
                  registry: Registry | None = None) -> list[dict[str, str]]:
    """Chat messages for prompted (zero-/few-shot) grounding."""
    msgs = [{"role": "system", "content": system_prompt(registry)}]
    for shot in shots:
        text, ir = (shot.text, shot.ir) if hasattr(shot, "ir") else shot  # type: ignore[union-attr]
        msgs.append({"role": "user", "content": f"Rule: {text}"})
        msgs.append({"role": "assistant", "content": ir_to_text(ir)})
    msgs.append({"role": "user", "content": f"Rule: {rule.strip()}"})
    return msgs
