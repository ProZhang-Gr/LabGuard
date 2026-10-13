"""Rule-based grounding baseline ("Regex BL", Table 2).

Keyword scores select the constraint type, a verb lexicon gives the task
tags, the registry alias scanner gives the substances, and a per-type
template fills the predicate with the numbers / zones / flags found in the
text.  The substance extractor (`extract_substances`) is also the regex
fallback used by LabGuard-Hybrid for the substance slot (Sec. 5.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from labguard.grounder.base import Grounder
from labguard.ir.registry import Registry
from labguard.ir.schema import LabGuardIR, sort_tags
from labguard.vocab import CONSTRAINT_TYPES

# ----------------------------------------------------------------------------
# Lexicons
# ----------------------------------------------------------------------------
TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "chemical_incompatibility": (r"\bincompatib", r"\b(mix|combin|react)\w*", r"\bkept apart\b|\bseparate\b",
                                 r"\bsame (vessel|container)\b|\btogether\b"),
    "storage_constraint": (r"\bstor(e|ed|age|ing)\b", r"\bcabinet\b|\bshelf\b", r"\bsegregat"),
    "temperature_constraint": (r"°\s*[CF]\b|\bdegrees? (celsius|fahrenheit)\b", r"\bheat(ing|ed)?\b|\btemperature\b",
                               r"\bhotter\b|\bwarm"),
    "container_capacity": (r"\bcapacity\b", r"\bfill(ed|ing)?\b|\boverfill", r"%|\bpercent\b|\bhalf\b|\bthirds?\b",
                           r"\b\d+(\.\d+)?\s*(mL|L|litres?|liters?)\b(?!\s*/)"),
    "velocity_limit": (r"\b\d+(\.\d+)?\s*(m|cm|mm)/s\b", r"\bspeed\b|\bvelocity\b", r"\bfast(er)?\b|\bslow"),
    "force_limit": (r"\b\d+(\.\d+)?\s*(N|newtons?)\b", r"\bforce\b", r"\bgrip|squeez"),
    "workspace_zone": (r"\bworkspace\b", r"\bzone\b|\bboundar|\bwork area\b", r"\b(stay|remain|keep)\w* (inside|within)\b"),
    "spill_prevention": (r"\bspill|splash", r"\bm?L/(s|min)\b|\bpour(ing)? rate\b", r"\btilt"),
    "equipment_check": (r"\b(check|verify|confirm|inspect|ensure)\w*\b",
                        r"\b(switched on|turned on|intact|dry|balanced|calibrated|clean|cracked|wet|labell?ed)\b"),
    "sequence_constraint": (r"\bbefore\b|\bafter\b|\buntil\b|\bfirst\b|\bthen\b|\bprecede", r"\bcap(ped)?\b|\blid\b"),
    "exposure_limit": (r"\binhal|vapou?rs?\b|\bbreath|\bexposure|\bfumes\b", r"\bH33\d\b|\bP26\d\b|\bP284\b",
                       r"\b\d+(\.\d+)?\s*(minutes?|min|seconds?|s)\b(?!/)"),
    "ventilation_requirement": (r"\bfume (hood|cupboard)|\bventilat|\bhood\b", r"\bP271\b"),
    "fire_safety": (r"\bflammab|\bignit|\bfire\b|\bspark|\bflame|\bheat source", r"\bH22\d\b|\bP210\b"),
    "ppe_requirement": (r"\bgloves?\b|\bgoggles\b|\blab coat\b|\bface shield\b|\brespirator\b",
                        r"\bPPE\b|\bprotective\b|\bwear\b"),
    "waste_handling": (r"\bwaste\b", r"\bdispos|\bdiscard|\bdrain\b|\bsink\b"),
}

TAG_PATTERNS: dict[str, str] = {
    "pick": r"\bpick\w*\b|\bgrasp|\bgrab|\blift",
    "place": r"\bplac(e|es|ing|ed)\b|\bset (it )?down\b|\bput\w* (it )?(down|on|in)\b",
    "press": r"\bpress(ing|es)?\b|\bbutton",
    "shake": r"\bshak",
    "stir": r"\bstir",
    "open": r"\bopen(s|ing|ed)?\b|\buncap",
    "close": r"\bclos(e|es|ing)\b|\breseal|\brecap",
    "pour": r"\bpour",
    "transfer": r"\btransfer",
    "mix": r"\bmix|\bcombin|\badd(ing|ed)?\b",
    "heat": r"\bheat|\bwarm|\bboil",
    "cool": r"\bcool|\bchill",
    "transport": r"\bcarr(y|ies|ying)|\btransport",
    "clean": r"\bclean(ing)?\b|\bwash|\brins",
    "dispense": r"\bdispens|\bpipett",
    "weigh": r"\bweigh",
    "centrifuge": r"\bcentrifug(e|ing)\b(?! (is|must|lid))|\bspin",
    "store": r"\bstor(e|es|ing|age|ed)\b",
    "dispose": r"\bdispos|\bdiscard",
    "move": r"\bmov(e|es|ing)\b|\bmotion",
    "inspect": r"\binspect|\bexamin",
}

DEFAULT_SCOPE: dict[str, tuple[str, ...]] = {
    "chemical_incompatibility": ("pour", "transfer", "mix"),
    "storage_constraint": ("place", "store"),
    "temperature_constraint": ("heat",),
    "container_capacity": ("pour", "transfer", "dispense"),
    "velocity_limit": ("transport",),
    "force_limit": ("pick",),
    "workspace_zone": ("transport", "move"),
    "spill_prevention": ("pour", "transfer"),
    "equipment_check": ("heat",),
    "sequence_constraint": ("pour",),
    "exposure_limit": ("open", "pour", "transfer"),
    "ventilation_requirement": ("open", "pour", "transfer", "mix"),
    "fire_safety": ("place", "transport", "store"),
    "ppe_requirement": ("pour", "transfer"),
    "waste_handling": ("dispose",),
}

HIGH_CUES = re.compile(r"\b(never|always|must|shall|mandatory|strictly|under no circumstances|prohibited|forbidden|"
                       r"only|danger|fatal|severe|violent\w*|explo\w+)\b", re.IGNORECASE)
LOW_CUES = re.compile(r"\b(where possible|preferabl[ey]|good practice|try not|try to|ideally|recommended|"
                      r"if possible)\b", re.IGNORECASE)
MODIFY_CUES = re.compile(r"\b(at most|no more than|not exceed|exceed\w*|above|below|limit\w*|clamp\w*|maximum|"
                         r"slow\w*|faster|higher|at least|keep\w* .* (inside|within)|less than|cap)\b",
                         re.IGNORECASE)
LOWER_BOUND = re.compile(r"\b(at least|no less than|minimum|or more|more than|further than|farther than)\b",
                         re.IGNORECASE)

FLAG_WORDS: dict[str, tuple[str, ...]] = {  # flag -> surface cues (antonyms map to the positive flag)
    "on": (r"switched on", r"turned on", r"running", r"\bon\b", r"operating", r"switched off", r"turned off",
           r"\boff\b"),
    "calibrated": (r"calibrat",),
    "clean": (r"\bclean\b", r"\bdirty\b", r"contaminat"),
    "sash_lowered": (r"\bsash\b",),
    "balanced": (r"\bbalanc", r"unbalanc"),
    "lid_closed": (r"\blid\b",),
    "intact": (r"\bintact\b", r"\bcrack", r"\bchipped\b", r"\bdamaged\b", r"\bbroken\b"),
    "dry": (r"\bdry\b", r"\bwet\b", r"\bdamp\b"),
    "labeled": (r"\blabell?ed\b", r"\bunlabell?ed\b", r"\blabel\b"),
    "closed": (r"\bclosed\b", r"\bcapped\b"),
    "not_full": (r"\bfull\b",),
}
STEP_WORDS: dict[str, str] = {
    "label": r"\blabel", "balance": r"\bbalanc", "calibrate": r"\bcalibrat", "rinse": r"\brins",
    "vent": r"\bvent(ing|ed)?\b", "preheat": r"\bpre-?heat", "neutralize": r"\bneutrali[sz]", "inspect": r"\binspect",
    "close": r"\bclos(e|ed|ing)\b", "open": r"\bopen(ed|ing)?\b", "cool": r"\bcool", "clean": r"\bclean",
}
PPE_WORDS: dict[str, str] = {
    "gloves": r"\bgloves?\b", "goggles": r"\bgoggles\b|\beye protection\b|\bsafety glasses\b",
    "lab_coat": r"\blab coat\b", "face_shield": r"\bface shield\b", "respirator": r"\brespirator|\brespiratory\b",
}
WASTE_WORDS: dict[str, str] = {
    "halogenated_waste": r"\bhalogenated\b", "organic_waste": r"\b(non-halogenated|organic|solvent) waste\b",
    "aqueous_waste": r"\baqueous waste\b", "acid_waste": r"\bacid(ic)? waste\b", "base_waste": r"\b(base|basic|alkaline) waste\b",
    "oxidizer_waste": r"\boxidi[sz]er waste\b|\boxidi[sz]ing waste\b",
}
ZONE_WORDS: dict[str, str] = {
    "fume_hood": r"\bfume (hood|cupboard)\b|\bhood\b", "workspace": r"\bworkspace\b|\bwork area\b",
    "hot_zone": r"\bhot zone\b|\bheating (area|zone)\b", "acid_cabinet": r"\bacid (storage )?cabinet\b",
    "base_cabinet": r"\b(base|alkali|corrosives?) (storage )?cabinet\b",
    "flammable_cabinet": r"\bflammables? (storage )?cabinet\b", "oxidizer_shelf": r"\boxidi[sz]er shelf\b",
    "general_shelf": r"\bgeneral shelf\b", "waste_area": r"\bwaste area\b", "sink": r"\bsink\b",
}
HAZARD_WORDS: dict[str, str] = {
    "flammable": r"\bflammab", "oxidizer": r"\boxidi[sz](er|ing)", "corrosive": r"\bcorrosive",
    "toxic": r"\btoxic", "water_reactive": r"\bwater[- ]reactive", "acid": r"\bacids\b", "base": r"\bbases\b",
}
# entity mentions that name a zone or a destination rather than a rule subject
NON_SUBJECT_ENTITIES: dict[str, tuple[str, ...]] = {
    "fume_hood": ("ventilation_requirement", "exposure_limit", "storage_constraint"),
    "waste_container": ("waste_handling", "chemical_incompatibility"),
}
OBJECT_TYPES = ("container_capacity", "velocity_limit", "force_limit")

_NUM = r"(\d+(?:\.\d+)?)"
UNITS: tuple[tuple[str, str, Any], ...] = (  # (pattern, quantity, converter)
    (_NUM + r"\s*°\s*F\b|" + _NUM + r"\s*degrees? fahrenheit", "temperature", lambda v: (v - 32) * 5 / 9),
    (_NUM + r"\s*°\s*C?\b|" + _NUM + r"\s*degrees? (celsius|C)\b", "temperature", lambda v: v),
    (_NUM + r"\s*mL\s*/\s*min\b", "rate", lambda v: v / 60),
    (_NUM + r"\s*L\s*/\s*min\b", "rate", lambda v: v * 1000 / 60),
    (_NUM + r"\s*mL\s*/\s*s\b", "rate", lambda v: v),
    (_NUM + r"\s*mm\s*/\s*s\b", "speed", lambda v: v / 1000),
    (_NUM + r"\s*cm\s*/\s*s\b", "speed", lambda v: v / 100),
    (_NUM + r"\s*m\s*/\s*s\b", "speed", lambda v: v),
    (_NUM + r"\s*(N|newtons?)\b", "force", lambda v: v),
    (_NUM + r"\s*(mL|millilit\w+)\b", "volume", lambda v: v),
    (_NUM + r"\s*(L|lit(re|er)s?)\b", "volume", lambda v: v * 1000),
    (_NUM + r"\s*%", "fraction", lambda v: v / 100),
    (_NUM + r"\s*(minutes?|min)\b", "time", lambda v: v * 60),
    (_NUM + r"\s*(seconds?|s)\b", "time", lambda v: v),
    (_NUM + r"\s*(cm|centimet\w+)\b", "distance", lambda v: v / 100),
    (_NUM + r"\s*(mm)\b", "distance", lambda v: v / 1000),
    (_NUM + r"\s*(m|met(re|er)s?)\b", "distance", lambda v: v),
    (_NUM + r"\s*(degrees?|°)", "angle", lambda v: v),
)
WORD_FRACTIONS = {"half": 0.5, "two-thirds": 0.67, "two thirds": 0.67, "three-quarters": 0.75,
                  "three quarters": 0.75, "a third": 0.33, "one third": 0.33}


def fmt_num(v: float) -> str:
    v = round(float(v), 3)
    return str(int(v)) if v == int(v) else f"{v:g}"


@dataclass
class Quantity:
    kind: str
    value: float
    span: tuple[int, int]


def extract_quantities(text: str) -> list[Quantity]:
    out: list[Quantity] = []
    taken: list[tuple[int, int]] = []
    for pat, kind, conv in UNITS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE if kind not in ("volume", "force") else 0):
            span = m.span()
            if any(not (span[1] <= a or span[0] >= b) for a, b in taken):
                continue
            raw = next(g for g in m.groups() if g is not None)
            taken.append(span)
            out.append(Quantity(kind, conv(float(raw)), span))
    for word, frac in WORD_FRACTIONS.items():
        for m in re.finditer(r"\b" + word + r"\b", text, flags=re.IGNORECASE):
            out.append(Quantity("fraction", frac, m.span()))
    out.sort(key=lambda q: q.span[0])
    return out


def first(qs: list[Quantity], *kinds: str) -> Quantity | None:
    return next((q for q in qs if q.kind in kinds), None)


def _bound_op(text: str) -> str:
    return ">=" if LOWER_BOUND.search(text) else "<="


# ----------------------------------------------------------------------------
# Field extractors (shared with the hybrid grounder)
# ----------------------------------------------------------------------------
def classify_type(text: str) -> str:
    scores = {t: sum(bool(re.search(p, text, flags=re.IGNORECASE)) for p in pats)
              for t, pats in TYPE_PATTERNS.items()}
    best = max(scores.values())
    if best == 0:
        return "equipment_check"
    # ties broken by corpus frequency (Table 7 order)
    return next(t for t in CONSTRAINT_TYPES if scores[t] == best)


def extract_tags(text: str, ctype: str | None = None) -> tuple[str, ...]:
    tags = [t for t, p in TAG_PATTERNS.items() if re.search(p, text, flags=re.IGNORECASE)]
    if not tags and ctype:
        tags = list(DEFAULT_SCOPE.get(ctype, ()))
    return sort_tags(tags)


def extract_severity(text: str) -> str:
    if LOW_CUES.search(text):
        return "low"
    return "high" if HIGH_CUES.search(text) else "medium"


def extract_intervention(text: str, ctype: str, severity: str) -> str:
    if severity == "low":
        return "log"
    numeric = ctype in ("temperature_constraint", "container_capacity", "velocity_limit", "force_limit",
                        "spill_prevention", "workspace_zone", "storage_constraint", "waste_handling")
    return "modify" if numeric and MODIFY_CUES.search(text) else "block"


def extract_substances(text: str, registry: Registry, ctype: str | None = None,
                       predicate: str | None = None) -> tuple[str, ...]:
    """Registry mentions, minus entities that only name a zone/destination for this type."""
    found = [sid for sid, _ in registry.find_mentions(text)]
    out = []
    for sid in found:
        if ctype and ctype in NON_SUBJECT_ENTITIES.get(sid, ()) and not (predicate and f"'{sid}'" in predicate
                                                                          and "status(" in predicate):
            continue
        if sid not in out:
            out.append(sid)
    if ctype in OBJECT_TYPES and "glassware" not in out and re.search(r"\b(container|vessel|bottle)s?\b", text, re.I):
        out.append("glassware")
    return tuple(registry.sort_ids(out))


def _match_key(table: dict[str, str], text: str) -> list[str]:
    return [k for k, p in table.items() if re.search(p, text, flags=re.IGNORECASE)]


def _flag_for(entity: str, text: str, registry: Registry) -> str | None:
    for flag in registry.equipment_flags.get(entity, ()):
        if any(re.search(p, text, flags=re.IGNORECASE) for p in FLAG_WORDS.get(flag, ())):
            return flag
    return None


def build_predicate(ctype: str, text: str, subs: tuple[str, ...], registry: Registry) -> str:
    qs = extract_quantities(text)
    op = _bound_op(text)
    chems = [s for s in subs if registry.is_chemical(s)]
    ents = [s for s in subs if registry.is_entity(s)]
    zones = _match_key(ZONE_WORDS, text)
    negated_zone = bool(re.search(r"\b(outside|leave|never enter|away from|not .* (in|into|inside))\b", text, re.I))

    if ctype in ("chemical_incompatibility", "storage_constraint"):
        if ctype == "storage_constraint" and zones:
            return f"in_zone('{zones[0]}')"
        hz = _match_key(HAZARD_WORDS, text)
        if len(chems) < 2 and len(hz) >= 2:
            return f"not (hazard(src, '{hz[0]}') and hazard(dst, '{hz[1]}'))"
        return "not incompatible(src, dst)"
    if ctype == "temperature_constraint":
        q = first(qs, "temperature")
        v = fmt_num(q.value) if q else "60"
        if re.search(r"\bpick|\bhotter\b|\bhandl|\bcool (down )?to\b", text, re.I) and not re.search(r"\bheat", text, re.I):
            return f"src.temperature <= {v}"
        return f"action.target_temperature {op} {v}"
    if ctype == "container_capacity":
        q = first(qs, "fraction", "volume")
        if q is None:
            return "action.fill_after <= 0.8"
        if q.kind == "volume":
            return f"action.volume <= {fmt_num(q.value)}"
        attr = "src.fill_ratio" if re.search(r"\b(shak|heat|carry|transport)", text, re.I) else "action.fill_after"
        return f"{attr} <= {fmt_num(q.value)}"
    if ctype == "velocity_limit":
        q = first(qs, "speed")
        attr = "robot.speed" if re.search(r"\b(joint|arm|robot) (speed|velocity)\b", text, re.I) else "action.speed"
        return f"{attr} <= {fmt_num(q.value) if q else '0.25'}"
    if ctype == "force_limit":
        fs = [q for q in qs if q.kind == "force"]
        if len(fs) >= 2:
            lo, hi = sorted((fs[0].value, fs[1].value))
            return f"action.grip_force >= {fmt_num(lo)} and action.grip_force <= {fmt_num(hi)}"
        return f"action.grip_force {op} {fmt_num(fs[0].value) if fs else '20'}"
    if ctype == "workspace_zone":
        q = first(qs, "distance")
        if q is not None and ents:
            return f"distance('{ents[0]}') >= {fmt_num(q.value)}"
        zone = next((z for z in zones if z != "workspace"), zones[0] if zones else "workspace")
        return f"not in_zone('{zone}')" if negated_zone and zone != "workspace" else f"in_zone('{zone}')"
    if ctype == "spill_prevention":
        q = first(qs, "rate", "angle", "speed")
        if q is None:
            return "action.pour_rate <= 25"
        attr = {"rate": "action.pour_rate", "angle": "action.tilt_angle", "speed": "action.speed"}[q.kind]
        return f"{attr} <= {fmt_num(q.value)}"
    if ctype == "equipment_check":
        checks = []
        for e in ents or ["glassware"]:
            flag = _flag_for(e, text, registry)
            if flag:
                checks.append(f"status('{e}', '{flag}')")
        return " and ".join(checks) if checks else f"status('{(ents or ['glassware'])[0]}', 'intact')"
    if ctype == "sequence_constraint":
        if re.search(r"\b(lid|cap)\b|\bcapped\b|\bsealed\b", text, re.I):
            closed = re.search(r"\b(close|seal|recap|cap)\w*\b.*\bbefore\b", text, re.I)
            return "not src.lid_open" if closed else "src.lid_open"
        steps = [s for s, p in STEP_WORDS.items() if re.search(p, text, re.I)]
        return f"done('{steps[0] if steps else 'inspect'}')"
    if ctype == "exposure_limit":
        q = first(qs, "time")
        if q is None:
            return "in_zone('fume_hood') or has_ppe('respirator')"
        attr = "src.open_time" if re.search(r"\bopen|uncapp", text, re.I) else "action.duration"
        return f"{attr} <= {fmt_num(q.value)}"
    if ctype == "ventilation_requirement":
        if re.search(r"\b(switched on|turned on|running|operating|is on)\b", text, re.I):
            return "status('fume_hood', 'on')"
        return "in_zone('fume_hood')"
    if ctype == "fire_safety":
        q = first(qs, "distance")
        ent = next((e for e in ents if e in ("hot_plate",)), None)
        if q is not None:
            target = ent or "hot_plate"
            pred = f"distance('{target}') >= {fmt_num(q.value)}"
            if re.search(r"\b(switched on|turned on|is on|running)\b", text, re.I):
                pred += f" or not status('{target}', 'on')"
            return pred
        t = first(qs, "temperature")
        if t is not None:
            return f"action.target_temperature <= {fmt_num(t.value)}"
        return "not in_zone('hot_zone')"
    if ctype == "ppe_requirement":
        items = _match_key(PPE_WORDS, text) or ["gloves"]
        return " and ".join(f"has_ppe('{i}')" for i in items)
    if ctype == "waste_handling":
        streams = _match_key(WASTE_WORDS, text)
        if not streams and chems:
            streams = [registry.waste_stream(chems[0]) or "aqueous_waste"]
        return f"dst.label == '{streams[0] if streams else 'aqueous_waste'}'"
    return "True"


class RegexGrounder(Grounder):
    name = "regex"

    def ground(self, text: str) -> LabGuardIR:
        ctype = classify_type(text)
        subs = extract_substances(text, self.registry, ctype)
        pred = build_predicate(ctype, text, subs, self.registry)
        subs = extract_substances(text, self.registry, ctype, pred)
        sev = extract_severity(text)
        return LabGuardIR(ctype, subs, pred, extract_tags(text, ctype), sev, extract_intervention(text, ctype, sev))
