"""Seed-rule templates for the LabGuard-Bench reconstruction (Sec. 3.3, Appendix D).

The paper does not release its 203 seed rules, so `labguard.bench` regenerates
a corpus with the published statistics (Table 7 type counts, Table 8 source
families, difficulty strata).  Each generator below samples one seed rule:
a gold LabGuard-IR instance plus several wordings in the style of its source
family, so that source-held-out evaluation measures phrasing / rule-structure
shift as in the paper:

  lab_manual       procedure-focused instructions   ("Before heating, check ...")
  pubchem_ghs      GHS hazard / precaution phrases   ("H225 ... P210: Keep away ...")
  expert           controller-facing robot rules     ("The robot must never ...")
  chem_safety_ref  pairwise incompatibility facts    ("A is incompatible with B")

`texts[0]` of a draft is the seed wording, the remaining texts are
alternative wordings that the paraphraser draws from.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from labguard.ir.predicate import Predicate
from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import LabGuardIR, sort_tags


class NoCandidate(RuntimeError):
    """The generator cannot satisfy the require/exclude constraints."""


@dataclass
class RuleDraft:
    ir: LabGuardIR
    texts: list[str]
    complexity: float
    template: str
    flags: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Wording helpers
# ----------------------------------------------------------------------------
# Severity cues (prefixes taking a bare infinitive).  Paraphrases keep the cue strength.
NEG = {
    "high": ("Never", "You must never", "Under no circumstances should you", "It is strictly forbidden to"),
    "medium": ("Do not", "You should not", "Please do not"),
    "low": ("Where possible, do not", "It is preferable not to", "Try not to"),
}
POS = {
    "high": ("Always", "You must always", "It is mandatory to"),
    "medium": ("Make sure to", "Be sure to", "You should"),
    "low": ("Where possible,", "It is good practice to", "Preferably,"),
}
RNEG = {
    "high": ("The robot must never", "The controller shall never", "Under no circumstances may the robot"),
    "medium": ("The robot must not", "The robot should not", "The controller should not"),
    "low": ("Where possible, the robot should not", "The robot should preferably not"),
}
RPOS = {
    "high": ("The robot must always", "The controller shall always"),
    "medium": ("The robot should", "The controller should"),
    "low": ("Where possible, the robot should", "The robot should preferably"),
}

# task tag -> (verb phrase, gerund phrase, noun)
TAG_VERB: dict[str, tuple[str, str, str]] = {
    "pick": ("pick up", "picking up", "pick"), "place": ("place", "placing", "place"),
    "press": ("press the start button", "pressing the start button", "button-press"),
    "shake": ("shake", "shaking", "shake"), "stir": ("stir", "stirring", "stir"),
    "open": ("open", "opening", "open"), "close": ("close", "closing", "close"),
    "pour": ("pour", "pouring", "pour"), "transfer": ("transfer", "transferring", "transfer"),
    "mix": ("mix", "mixing", "mixing"), "heat": ("heat", "heating", "heating"),
    "cool": ("cool", "cooling", "cooling"), "transport": ("carry", "carrying", "transport"),
    "clean": ("clean", "cleaning", "cleaning"), "dispense": ("dispense", "dispensing", "dispense"),
    "weigh": ("weigh", "weighing", "weighing"),
    "centrifuge": ("run the centrifuge", "running the centrifuge", "centrifuge"),
    "store": ("store", "storing", "storage"), "dispose": ("dispose of waste", "disposing of waste", "disposal"),
    "move": ("move", "moving", "motion"), "inspect": ("inspect", "inspecting", "inspection"),
}

GHS_H: dict[str, str] = {
    "H224": "Extremely flammable liquid and vapour.",
    "H225": "Highly flammable liquid and vapour.",
    "H226": "Flammable liquid and vapour.",
    "H260": "In contact with water releases flammable gases which may ignite spontaneously.",
    "H272": "May intensify fire; oxidiser.",
    "H290": "May be corrosive to metals.",
    "H301": "Toxic if swallowed.",
    "H302": "Harmful if swallowed.",
    "H314": "Causes severe skin burns and eye damage.",
    "H318": "Causes serious eye damage.",
    "H319": "Causes serious eye irritation.",
    "H331": "Toxic if inhaled.",
    "H335": "May cause respiratory irritation.",
    "H336": "May cause drowsiness or dizziness.",
    "H351": "Suspected of causing cancer.",
    "EUH031": "Contact with acids liberates toxic gas.",
}
GHS_CODES: dict[str, tuple[str, ...]] = {
    "HCl": ("H290", "H314", "H335"), "NaClO": ("H314", "EUH031"), "H2SO4": ("H290", "H314"),
    "HNO3": ("H272", "H314"), "NaOH": ("H290", "H314"), "NH3": ("H314", "H335"),
    "H2O2": ("H302", "H318"), "KMnO4": ("H272", "H302"), "Na": ("H260", "H314"),
    "C2H5OH": ("H225", "H319"), "CH3OH": ("H225", "H301", "H331"), "C3H6O": ("H225", "H319", "H336"),
    "C4H10O": ("H224", "H302", "H336"), "CHCl3": ("H302", "H331", "H351"), "CH3COOH": ("H226", "H314"),
}
INHALATION_CODES = ("H331", "H335", "H336")
# Severity of GHS-sourced rules follows the hazard statement they are derived from
GHS_SEVERITY: dict[str, str] = {
    "H224": "high", "H225": "high", "H226": "medium", "H260": "high", "H272": "high", "H290": "medium",
    "H301": "high", "H302": "medium", "H314": "high", "H318": "high", "H319": "medium", "H331": "high",
    "H335": "medium", "H336": "low", "H351": "high", "EUH031": "high",
}

WASTE_TEXT = {
    "halogenated_waste": "halogenated waste", "organic_waste": "organic solvent waste",
    "aqueous_waste": "aqueous waste", "acid_waste": "acid waste", "base_waste": "base waste",
    "oxidizer_waste": "oxidizer waste",
}
CABINET_TEXT = {
    "acid_cabinet": "acid cabinet", "base_cabinet": "base cabinet",
    "flammable_cabinet": "flammables cabinet", "oxidizer_shelf": "oxidizer shelf",
}
CABINET_CLASS = {"acid_cabinet": "acid", "base_cabinet": "base", "flammable_cabinet": "flammable",
                 "oxidizer_shelf": "oxidizer"}
PPE_TEXT = {"gloves": "protective gloves", "goggles": "safety goggles", "face_shield": "a face shield",
            "lab_coat": "a lab coat", "respirator": "respiratory protection"}

SPECIAL_REACTIONS: dict[frozenset[str], str] = {
    frozenset({"HCl", "NaClO"}): "releases toxic chlorine gas",
    frozenset({"NaClO", "NH3"}): "produces toxic chloramine vapours",
    frozenset({"NaClO", "H2SO4"}): "liberates chlorine gas",
    frozenset({"NaClO", "HNO3"}): "liberates chlorine gas",
    frozenset({"NaClO", "CH3COOH"}): "liberates chlorine gas",
    frozenset({"Na", "H2O"}): "liberates hydrogen that can ignite explosively",
    frozenset({"HNO3", "C2H5OH"}): "can react explosively",
    frozenset({"H2O2", "C3H6O"}): "can form explosive peroxides",
    frozenset({"KMnO4", "H2SO4"}): "forms an explosive oxidiser",
    frozenset({"NaClO", "C3H6O"}): "forms toxic chloroform",
    frozenset({"NaOH", "CHCl3"}): "can react violently",
}


def fmt(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:g}"


def join(items: Iterable[str], conj: str = "or") -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


def verbs(scope: Iterable[str], form: int = 0, conj: str = "or") -> str:
    return join([TAG_VERB[t][form] for t in sort_tags(scope)], conj)


_SENT_START = re.compile(r"(^|[.!?]\s+)([a-z])")


def tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = _SENT_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    return text if text.endswith((".", "!", "?")) else text + "."


def _n_atoms(node: ast.AST) -> int:
    if isinstance(node, ast.BoolOp):
        return sum(_n_atoms(v) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _n_atoms(node.operand)
    return 1


def structural_complexity(ir: LabGuardIR, registry: Registry) -> float:
    """Complexity from the gold IR alone (number of atoms, substances, scope breadth, ...)."""
    p = Predicate(ir.predicate, registry)
    tree = p.canonical_tree
    c = 1.0 * max(0, _n_atoms(tree.body) - 1)
    c += 0.5 * max(0, len(ir.substances) - 2)
    c += 0.25 * max(0, len(ir.action_scope) - 3)
    if "hazard" in p.call_names():
        c += 1.0
    if any(isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and isinstance(n.operand, ast.BoolOp)
           for n in ast.walk(tree)):
        c += 0.5
    return c


# ----------------------------------------------------------------------------
# Generation context
# ----------------------------------------------------------------------------
class GenContext:
    def __init__(self, rng: np.random.Generator, registry: Registry | None = None,
                 require: str | None = None, exclude: Iterable[str] = ()) -> None:
        self.rng = rng
        self.reg = registry or default_registry()
        self.require = require
        self.exclude = frozenset(exclude)
        self.bonus = 0.0  # complexity added by the wording (aliases, unit conversions, implicit S)
        self.flags: dict[str, Any] = {}

    # -- sampling -------------------------------------------------------------
    def choice(self, seq: Iterable[Any]) -> Any:
        seq = list(seq)
        if not seq:
            raise NoCandidate("empty choice")
        return seq[int(self.rng.integers(len(seq)))]

    def coin(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def weighted(self, dist: dict[Any, float]) -> Any:
        keys = list(dist)
        p = np.array([dist[k] for k in keys], dtype=float)
        return keys[int(self.rng.choice(len(keys), p=p / p.sum()))]

    def shuffled(self, seq: Iterable[Any]) -> list[Any]:
        out = list(seq)
        self.rng.shuffle(out)
        return out

    def pick(self, pool: Iterable[str], k: int = 1) -> list[str]:
        """k distinct chemicals from `pool`, honouring require / exclude."""
        pool = [c for c in dict.fromkeys(pool) if c not in self.exclude]
        if len(pool) < k:
            raise NoCandidate(f"need {k} of {pool}")
        if self.require in pool:
            return [self.require] + self.shuffled(c for c in pool if c != self.require)[:k - 1]
        return self.shuffled(pool)[:k]

    def chems_with(self, *hazards: str, volatile: bool | None = None) -> list[str]:
        out = []
        for sid in self.reg.chemical_ids:
            info = self.reg.substances[sid]
            if hazards and not any(h in info.hazards for h in hazards):
                continue
            if volatile is not None and info.volatile != volatile:
                continue
            out.append(sid)
        return out

    def pairs(self) -> list[tuple[str, str]]:
        out = sorted(tuple(self.reg.sort_ids(p)) for p in self.reg.incompatible_pairs)
        out = [p for p in out if not set(p) & self.exclude]
        if self.require is not None and any(self.require in p for p in out):
            out = [p for p in out if self.require in p]
        return out  # type: ignore[return-value]

    # -- surface forms --------------------------------------------------------
    def name(self, sid: str, alias_p: float = 0.2) -> str:
        names = [n for n in self.reg.surface_names(sid) if "_" not in n]
        if len(names) > 1 and self.coin(alias_p):
            self.bonus += 0.5
            self.flags["alias"] = True
            return self.choice(names[1:])
        return names[0]

    def ghs(self, sid: str, prefer: tuple[str, ...] = ()) -> tuple[str, str]:
        codes = GHS_CODES.get(sid, ("H302",))
        preferred = [c for c in codes if c in prefer]
        code = self.choice(preferred or codes)
        return code, GHS_H[code]

    def temp(self, t: float) -> str:
        if self.coin(0.15):
            self.bonus += 1.5
            self.flags["unit_conversion"] = "F"
            return f"{fmt(t * 9 / 5 + 32)} °F"
        return self.choice((f"{fmt(t)} °C", f"{fmt(t)} degrees Celsius", f"{fmt(t)}°C"))

    def speed(self, v: float) -> str:
        r = self.rng.random()
        if r < 0.25:
            self.bonus += 1.0
            self.flags["unit_conversion"] = "cm/s"
            return f"{fmt(round(v * 100, 3))} cm/s"
        if r < 0.35:
            self.bonus += 1.0
            self.flags["unit_conversion"] = "mm/s"
            return f"{fmt(round(v * 1000, 3))} mm/s"
        return f"{fmt(v)} m/s"

    def dist(self, d: float) -> str:
        if self.coin(0.2):
            return f"{fmt(d)} m"
        self.bonus += 0.5
        return f"{fmt(round(d * 100, 3))} cm"

    def seconds(self, s: float) -> str:
        if s % 60 == 0 and self.coin(0.4):
            self.bonus += 1.0
            self.flags["unit_conversion"] = "min"
            m = int(s // 60)
            return "one minute" if m == 1 else f"{m} minutes"
        return self.choice((f"{fmt(s)} s", f"{fmt(s)} seconds"))

    def fraction(self, f: float) -> str:
        words = {0.5: "half", 0.75: "three-quarters"}
        if f in words and self.coin(0.5):
            self.bonus += 1.0
            return words[f]
        return f"{fmt(round(f * 100, 3))}%"

    def volume(self, v: float) -> str:
        if v >= 100 and self.coin(0.25):
            self.bonus += 1.0
            self.flags["unit_conversion"] = "L"
            return f"{fmt(v / 1000)} L"
        return f"{fmt(v)} mL"

    def rate(self, r: float) -> str:
        if self.coin(0.15):
            self.bonus += 1.5
            self.flags["unit_conversion"] = "L/min"
            return f"{fmt(r * 60 / 1000)} L/min"
        return f"{fmt(r)} mL/s"

    def reaction(self, a: str, b: str) -> str:
        special = SPECIAL_REACTIONS.get(frozenset({a, b}))
        if special:
            return special
        ha, hb = self.reg.hazards(a), self.reg.hazards(b)
        if "water_reactive" in ha | hb:
            return "reacts violently and releases flammable hydrogen"
        if ("oxidizer" in ha and "flammable" in hb) or ("oxidizer" in hb and "flammable" in ha):
            return "may cause fire or explosion"
        if ("acid" in ha and "base" in hb) or ("acid" in hb and "base" in ha):
            return "reacts violently and releases a lot of heat"
        return self.choice(("can react dangerously", "may trigger a hazardous reaction"))

    # -- output ---------------------------------------------------------------
    def render(self, table: dict[str, list[str]], style: str, sev: str, **fields: Any) -> list[str]:
        tpls = table.get(style) or table.get("*") or next(iter(table.values()))
        out = []
        for t in self.shuffled(tpls):
            f = dict(fields)
            f.update(neg=self.choice(NEG[sev]), pos=self.choice(POS[sev]),
                     rneg=self.choice(RNEG[sev]), rpos=self.choice(RPOS[sev]))
            out.append(tidy(t.format(**f)))
        return list(dict.fromkeys(out))

    def draft(self, ctype: str, subs: Iterable[str], pred: str, scope: Iterable[str], sev: str, intv: str,
              texts: list[str], template: str) -> RuleDraft:
        if sev == "low":
            intv = "log"
        ir = LabGuardIR(ctype, tuple(self.reg.sort_ids(subs)), pred, sort_tags(scope), sev, intv)
        return RuleDraft(ir, texts, structural_complexity(ir, self.reg) + self.bonus, template, dict(self.flags))


def _sev(ctx: GenContext, dist: dict[str, float]) -> str:
    return ctx.weighted(dist)


# ----------------------------------------------------------------------------
# 1. chemical_incompatibility  (chem_safety_ref, expert)
# ----------------------------------------------------------------------------
_INC_T = {
    "chem_safety_ref": [
        "{A} is incompatible with {B}; contact {rx}.",
        "Incompatible materials for {A}: {Band}. {neg} {act} them.",
        "{A} and {B} must be kept apart because mixing them {rx}.",
        "Reactivity hazard: {A} combined with {B} {rx}. {neg} {act} the two.",
        "{neg} {act} {A} with {B}; the mixture {rx}.",
    ],
    "expert": [
        "{rneg} {act} {A} with {B}; the combination {rx}.",
        "{rneg} let {A} and {B} end up in the same vessel.",
        "Reject any action that would {act} {A} with {B}.",
        "{rneg} {act} {A} together with {B}, because this {rx}.",
    ],
    "*": [
        "{neg} {act} {A} with {B}; this {rx}.",
        "{A} must never be combined with {B} because it {rx}.",
        "Keep {A} and {B} apart: mixing them {rx}.",
    ],
}
_INC_DISPOSE_T = {
    "*": [
        "{neg} discard {A} into a waste container that holds {B}; this {rx}.",
        "{A} waste is incompatible with {B} waste. Keep them in separate waste containers.",
        "{rneg} dispose of {A} in a waste container holding {B}.",
        "When disposing of {A}, make sure the waste container does not contain {B}, since contact {rx}.",
    ],
}
_INC_DIR_T = {
    "chem_safety_ref": [
        "{A} must never be added to {B}: it {rx}.",
        "Adding {A} to {B} {rx}. {neg} do it.",
        "{neg} add {A} to {B}; the addition {rx}.",
    ],
    "expert": [
        "{rneg} pour {A} into a container holding {B}.",
        "Block any pour or transfer whose source holds {A} and whose destination holds {B}.",
        "{rneg} transfer {A} into a vessel that already contains {B}; this {rx}.",
    ],
    "*": ["{neg} pour {A} into {B}; it {rx}.", "{A} must not be added to {B} because it {rx}."],
}
_CLASS_PAIRS = (
    ("oxidizer", "flammable", "oxidizers", "flammable solvents", "may cause fire or explosion"),
    ("acid", "base", "strong acids", "strong bases", "reacts violently and releases a lot of heat"),
)


def gen_chemical_incompatibility(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"high": 0.85, "medium": 0.15})
    kind = ctx.weighted({"pair": 0.6, "group": 0.15, "directional": 0.15, "class": 0.1})
    transfer4 = ("pour", "transfer", "mix", "dispense")
    if kind == "class":
        h1, h2, d1, d2, rx = ctx.choice(_CLASS_PAIRS)
        ctx.bonus += 1.0
        pred = (f"not (hazard(src, '{h1}') and hazard(dst, '{h2}')) and "
                f"not (hazard(src, '{h2}') and hazard(dst, '{h1}'))")
        texts = ctx.render(_INC_T, style, sev, A=d1, B=d2, Band=d2, act="mix", rx=rx)
        return ctx.draft("chemical_incompatibility", [], pred, transfer4, sev, "block", texts, "inc_class")
    if kind == "group":
        anchors = [c for c in ctx.reg.chemical_ids if c not in ctx.exclude
                   and len(ctx.reg.inc(c) - ctx.exclude) >= 3]
        if ctx.require in anchors and ctx.coin(0.5):
            a = ctx.require
        else:
            a = ctx.choice(anchors)
        partners = ctx.pick(ctx.reg.sort_ids(ctx.reg.inc(a)), k=int(ctx.choice((2, 3))))
        partners = ctx.reg.sort_ids(partners)
        A, Bs = ctx.name(a), [ctx.name(b) for b in partners]
        texts = ctx.render(_INC_T, style, sev, A=A, B=join(Bs, "or"), Band=join(Bs, "and"), act="mix",
                           rx=ctx.reaction(a, partners[0]))
        return ctx.draft("chemical_incompatibility", [a, *partners], "not incompatible(src, dst)", transfer4,
                         sev, "block", texts, "inc_group")
    a, b = ctx.choice(ctx.pairs())
    if ctx.coin(0.5):
        a, b = b, a
    A, B, rx = ctx.name(a), ctx.name(b), ctx.reaction(a, b)
    if kind == "directional":
        pred = f"not (contains(src, '{a}') and contains(dst, '{b}'))"
        texts = ctx.render(_INC_DIR_T, style, sev, A=A, B=B, rx=rx)
        return ctx.draft("chemical_incompatibility", [a, b], pred, ("pour", "transfer"), sev, "block", texts,
                         "inc_directional")
    scope_kind = ctx.weighted({"generic": 0.7, "explicit": 0.15, "dispose": 0.15})
    if scope_kind == "dispose":
        texts = ctx.render(_INC_DISPOSE_T, style, sev, A=A, B=B, rx=rx)
        return ctx.draft("chemical_incompatibility", [a, b], "not incompatible(src, dst)", ("dispose",), sev,
                         "block", texts, "inc_dispose")
    scope = transfer4 if scope_kind == "generic" else ("pour", "transfer", "mix")
    act = ctx.choice(("mix", "combine")) if scope_kind == "generic" else "pour, transfer or mix"
    texts = ctx.render(_INC_T, style, sev, A=A, B=B, Band=B, act=act, rx=rx)
    return ctx.draft("chemical_incompatibility", [a, b], "not incompatible(src, dst)", scope, sev, "block",
                     texts, f"inc_pair_{scope_kind}")


# ----------------------------------------------------------------------------
# 2. storage_constraint  (chem_safety_ref, expert)
# ----------------------------------------------------------------------------
_STORE_CLASS_T = {
    "chem_safety_ref": [
        "Store {A} separately from {B}.",
        "{A} must not be stored next to {B}.",
        "Segregation rule: keep {A} and {B} on different shelves.",
    ],
    "expert": [
        "{rneg} place {A} next to {B} when putting reagents away.",
        "When storing reagents, the robot must keep {A} away from {B}.",
    ],
}
_STORE_CAB_T = {
    "chem_safety_ref": [
        "{A} must be stored in the {cab}.",
        "Keep {A} only in the {cab}.",
        "Storage: {A} belongs in the {cab}.",
    ],
    "expert": [
        "{rpos} return {A} to the {cab} when storing it.",
        "The robot may only store {A} inside the {cab}.",
        "When putting {A} away, the controller must route it to the {cab}.",
    ],
}
_STORE_PAIR_T = {
    "chem_safety_ref": [
        "{neg} store {A} next to {B}.",
        "{A} and {B} must be stored apart; accidental contact {rx}.",
        "Storage incompatibility: {A} with {B}. Keep them on separate shelves.",
    ],
    "expert": [
        "{rneg} place {A} beside {B} on a storage shelf.",
        "{rneg} store {A} next to {B}, since contact {rx}.",
    ],
}


def gen_storage_constraint(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"high": 0.5, "medium": 0.5})
    kind = ctx.weighted({"class": 0.3, "cabinet": 0.4, "pair": 0.3})
    if kind == "class":
        h1, h2, d1, d2, _ = ctx.choice(_CLASS_PAIRS)
        ctx.bonus += 1.0
        pred = (f"not (hazard(src, '{h1}') and hazard(dst, '{h2}')) and "
                f"not (hazard(src, '{h2}') and hazard(dst, '{h1}'))")
        texts = ctx.render(_STORE_CLASS_T, style, sev, A=d1, B=d2)
        return ctx.draft("storage_constraint", [], pred, ("place", "store"), sev, "modify", texts, "store_class")
    if kind == "cabinet":
        cab = ctx.choice(CABINET_TEXT)
        members = [c for c in ctx.chems_with(CABINET_CLASS[cab]) if c not in ctx.exclude]
        chems = ctx.reg.sort_ids(ctx.pick(members, k=min(len(members), int(ctx.choice((1, 1, 2, 3))))))
        texts = ctx.render(_STORE_CAB_T, style, sev, A=join([ctx.name(c) for c in chems], "and"),
                           cab=CABINET_TEXT[cab])
        return ctx.draft("storage_constraint", chems, f"in_zone('{cab}')", ("store",), sev, "modify", texts,
                         "store_cabinet")
    a, b = ctx.choice(ctx.pairs())
    texts = ctx.render(_STORE_PAIR_T, style, sev, A=ctx.name(a), B=ctx.name(b), rx=ctx.reaction(a, b))
    return ctx.draft("storage_constraint", [a, b], "not incompatible(src, dst)", ("place", "store"), sev, "modify",
                     texts, "store_pair")


# ----------------------------------------------------------------------------
# 3. equipment_check  (expert, lab_manual)
# ----------------------------------------------------------------------------
# entity, flag, candidate scopes, "good" clause, "bad" noun phrase
_EQUIP = (
    ("hot_plate", "calibrated", (("heat",),), "the hot plate is calibrated", "an uncalibrated hot plate"),
    ("hot_plate", "clean", (("heat",),), "the hot plate surface is clean", "a contaminated hot plate"),
    ("fume_hood", "on", (("open", "pour"), ("heat", "mix")), "the fume hood is switched on",
     "a fume hood that is switched off"),
    ("fume_hood", "sash_lowered", (("heat", "mix"), ("pour",)), "the fume hood sash is lowered",
     "the fume hood sash raised"),
    ("centrifuge", "balanced", (("centrifuge",), ("centrifuge", "press")), "the centrifuge rotor is balanced",
     "an unbalanced centrifuge"),
    ("centrifuge", "lid_closed", (("centrifuge",), ("centrifuge", "press")), "the centrifuge lid is closed",
     "the centrifuge lid open"),
    ("glassware", "intact", (("heat",), ("pour", "transfer"), ("pick",)), "the glassware is free of cracks and chips",
     "cracked or chipped glassware"),
    ("glassware", "clean", (("pour", "transfer", "mix"),), "the glassware is clean", "dirty glassware"),
    ("glassware", "dry", (("pour", "transfer"),), "the glassware is completely dry", "wet glassware"),
)
_EQUIP_COMPOUND = (
    (("centrifuge", "balanced"), ("centrifuge", "lid_closed"), ("centrifuge",),
     "the centrifuge is balanced and its lid is closed", "an unbalanced or open centrifuge"),
    (("glassware", "intact"), ("glassware", "clean"), ("heat",),
     "the glassware is intact and clean", "cracked or dirty glassware"),
    (("hot_plate", "calibrated"), ("glassware", "intact"), ("heat",),
     "the hot plate is calibrated and the glassware is intact", "an uncalibrated hot plate or cracked glassware"),
    (("fume_hood", "on"), ("fume_hood", "sash_lowered"), ("heat", "mix"),
     "the fume hood is running with its sash lowered", "a fume hood that is off or has its sash raised"),
)
_EQUIP_T = {
    "lab_manual": [
        "Before {ger}{obj}, check that {good}.",
        "{pos} verify that {good} before you {verb}{obj}.",
        "{neg} {verb}{obj} using {bad}.",
        "Prior to {ger}{obj}, confirm that {good}.",
    ],
    "expert": [
        "{rpos} verify that {good} before executing any {noun} action{inv}.",
        "{rneg} start a {noun} action{inv} with {bad}.",
        "Precondition for {ger}{obj}: {good}.",
        "Abort the {noun} action{inv} if it cannot be confirmed that {good}.",
    ],
}


def gen_equipment_check(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"high": 0.5, "medium": 0.4, "low": 0.1})
    if ctx.coin(0.15):
        (e1, f1), (e2, f2), scope, good, bad = ctx.choice(_EQUIP_COMPOUND)
        pred = f"status('{e1}', '{f1}') and status('{e2}', '{f2}')"
        subs = [e1, e2]
        template = "equip_compound"
    else:
        e1, f1, scopes, good, bad = ctx.choice(_EQUIP)
        scope = ctx.choice(scopes)
        pred = f"status('{e1}', '{f1}')"
        subs = [e1]
        template = "equip_single"
    obj = inv = ""  # optional chemical context ("before heating ethanol, ...")
    if ctx.coin(0.3 if ctx.require is None else 0.9) and set(scope) <= {"heat", "pour", "transfer", "mix"}:
        pool = ctx.chems_with("flammable", "corrosive", "toxic", "water_reactive")
        if (e1, f1) == ("glassware", "dry"):
            pool = ctx.chems_with("water_reactive")
        chem = ctx.pick(pool)[0]
        subs.append(chem)
        A = ctx.name(chem)
        obj, inv = f" {A}", f" involving {A}"
    texts = ctx.render(_EQUIP_T, style, sev, good=good, bad=bad, ger=verbs(scope, 1), verb=verbs(scope, 0),
                       noun=verbs(scope, 2), obj=obj, inv=inv)
    return ctx.draft("equipment_check", subs, pred, scope, sev, "block", texts, template)


# ----------------------------------------------------------------------------
# 4. sequence_constraint  (expert, lab_manual)
# ----------------------------------------------------------------------------
# key -> (substance pool spec, predicate, scope, severity dist, templates)
_SEQ_OPTIONS: dict[str, dict[str, Any]] = {
    "lid_pour": dict(pred="src.lid_open", scope=("pour",), chems="volatile", t={
        "lab_manual": ["{pos} remove the cap before pouring from a {obj}.",
                       "{neg} pour from a {obj} whose lid is still closed.",
                       "Open the lid of the {obj} first, then pour.",
                       "Uncap the {obj} before you pour from it."],
        "expert": ["{rneg} execute a pour while the {obj} is still capped.",
                   "Precondition for pouring: the {obj} lid must be open.",
                   "{rpos} open the {obj} before pouring from it."]}),
    "lid_shake": dict(pred="not src.lid_open", scope=("shake",), chems=None, t={
        "lab_manual": ["{pos} close the lid before shaking a container.",
                       "{neg} shake a container with an open lid.",
                       "Cap the vessel tightly before you shake it."],
        "expert": ["{rneg} shake an uncapped container.",
                   "Precondition for shaking: the container lid must be closed.",
                   "{rpos} close the lid before executing a shake."]}),
    "sealed_heat": dict(pred="src.lid_open", scope=("heat",), chems=None, t={
        "lab_manual": ["{neg} heat a sealed container.",
                       "{pos} remove the lid before heating a container.",
                       "Closed vessels must not be heated; open them first."],
        "expert": ["{rneg} heat a container that is still sealed.",
                   "Precondition for heating: the vessel lid must be open.",
                   "{rpos} open the vessel before a heating action."]}),
    "acid_water": dict(pred="not (contains(src, 'H2O') and contains(dst, '{a}'))", scope=("pour", "transfer", "mix"),
                       chems=("H2SO4", "HNO3"), with_water=True, sev={"high": 1.0}, t={
        "lab_manual": ["{pos} add {A} to water, never water to {A}.",
                       "{neg} pour water into concentrated {A}.",
                       "When diluting {A}, add the acid slowly to the water and not the reverse."],
        "expert": ["{rneg} pour water into a container holding {A}.",
                   "Block any transfer whose source contains water and whose destination contains {A}.",
                   "{rpos} add {A} to water rather than water to {A}."]}),
    "balance": dict(pred="done('balance')", scope=("centrifuge",), entity="centrifuge", t={
        "lab_manual": ["{pos} balance the tubes before starting the centrifuge.",
                       "{neg} run the centrifuge before the rotor has been balanced.",
                       "Balance first, then start the centrifuge."],
        "expert": ["{rneg} run the centrifuge before the balance step is complete.",
                   "The balance step must precede every centrifuge run."]}),
    "label_store": dict(pred="done('label')", scope=("store",), chems=None, t={
        "lab_manual": ["{pos} label every container before storing it.",
                       "{neg} store a container before it has been labeled.",
                       "Containers must be labeled before they are put away."],
        "expert": ["{rneg} store a container before the label step has been completed.",
                   "The label step must precede any storage action."]}),
    "neutralize": dict(pred="done('neutralize')", scope=("dispose",), chems="acid_base", t={
        "lab_manual": ["{pos} neutralize {A} waste before disposing of it.",
                       "{neg} dispose of {A} solutions before they have been neutralized.",
                       "Neutralize {A} first; only then may it be discarded."],
        "expert": ["{rneg} dispose of {A} until the neutralize step has been completed.",
                   "Disposal of {A} requires a completed neutralization step."]}),
    "vent": dict(pred="done('vent')", scope=("open",), chems=("C4H10O",), t={
        "lab_manual": ["{pos} vent a vessel containing shaken {A} before opening it.",
                       "{neg} open a vessel of shaken {A} before it has been vented."],
        "expert": ["{rneg} open a container of {A} before the vent step is complete."]}),
    "calibrate": dict(pred="done('calibrate')", scope=("weigh", "dispense"), chems=None, t={
        "lab_manual": ["{pos} calibrate the balance before weighing or dispensing reagents.",
                       "{neg} weigh or dispense reagents on a balance that has not been calibrated."],
        "expert": ["{rneg} weigh or dispense before the calibrate step is complete."]}),
    "stir_heat": dict(pred="done('stir')", scope=("heat",), chems=None, t={
        "lab_manual": ["{pos} start stirring before heating a liquid to prevent bumping.",
                       "{neg} heat a liquid before stirring has started; it may boil violently."],
        "expert": ["{rneg} heat a liquid before a stir action has been executed.",
                   "A stir action must precede heating to avoid violent boiling."]}),
    "close_transport": dict(pred="not src.lid_open", scope=("transport",), chems="volatile_opt", t={
        "lab_manual": ["{pos} close {obj_pl} before carrying them.",
                       "{neg} carry an open {obj}.",
                       "Cap the {obj} before you carry it across the bench."],
        "expert": ["{rneg} transport an uncapped {obj}.",
                   "Precondition for transport: the {obj} must be closed."]}),
    "both_open": dict(pred="src.lid_open and dst.lid_open", scope=("pour", "transfer"), chems=None, t={
        "lab_manual": ["{pos} uncap both the source and the receiving vessel before pouring or transferring.",
                       "{neg} pour or transfer unless both containers have been opened."],
        "expert": ["{rneg} pour or transfer while either the source or the destination is capped."]}),
    "label_dispose": dict(pred="done('label')", scope=("dispose",), entity="waste_container", t={
        "lab_manual": ["{pos} label the waste container before adding waste to it.",
                       "{neg} add waste to a waste container before it has been labeled."],
        "expert": ["{rneg} dispose of waste before the label step for the waste container is done."]}),
    "inspect_heat": dict(pred="done('inspect')", scope=("heat",), entity="glassware", t={
        "lab_manual": ["{pos} inspect glassware before heating it.",
                       "Heating must be preceded by a visual inspection of the glassware."],
        "expert": ["{rneg} heat glassware that has not been inspected first."]}),
    "open_pour": dict(pred="done('open')", scope=("pour",), chems=None, t={
        "lab_manual": ["Open the container first; pouring may only follow the opening step."],
        "expert": ["The open step must be completed before any pour.",
                   "{rneg} pour before the container has been opened."]}),
}


def gen_sequence_constraint(ctx: GenContext, style: str) -> RuleDraft:
    key = ctx.choice(_SEQ_OPTIONS)
    opt = _SEQ_OPTIONS[key]
    sev = _sev(ctx, opt.get("sev", {"high": 0.35, "medium": 0.55, "low": 0.1}))
    subs: list[str] = []
    fields: dict[str, Any] = {"obj": "reagent bottle", "obj_pl": "containers", "A": ""}
    pred = opt["pred"]
    spec = opt.get("chems")
    chem = None
    if isinstance(spec, tuple):
        chem = ctx.pick(spec)[0]
    elif spec == "acid_base":
        chem = ctx.pick(ctx.chems_with("acid", "base"))[0]
    elif spec in ("volatile", "volatile_opt") and ctx.coin(0.4 if ctx.require is None else 0.95):
        chem = ctx.pick(ctx.chems_with(volatile=True))[0]
    if chem is not None:
        subs.append(chem)
        fields["A"] = ctx.name(chem)
        fields["obj"] = f"{fields['A']} bottle"
        fields["obj_pl"] = f"{fields['A']} bottles"
        pred = pred.format(a=chem)
    if opt.get("with_water"):
        subs.append("H2O")
    if opt.get("entity"):
        subs.append(opt["entity"])
    texts = ctx.render(opt["t"], style, sev, **fields)
    return ctx.draft("sequence_constraint", subs, pred, opt["scope"], sev, "block", texts, f"seq_{key}")


# ----------------------------------------------------------------------------
# 5. exposure_limit  (pubchem_ghs)
# ----------------------------------------------------------------------------
_EXPO_OPEN_T = {"*": [
    "{A}: {H} P233: Keep the container tightly closed; never leave it open for more than {T}.",
    "{code} ({A}): {H} Minimise vapour release: a container of {A} may stay open for at most {T}.",
    "{A} - {code}: {H} Keep containers closed except during transfer and never open for longer than {T}.",
    "P233 / {code} ({A}): the container must not remain open longer than {T}.",
]}
_EXPO_DUR_T = {"*": [
    "{A}: {H} Limit each {noun} operation to {T}.",
    "P260 ({A}): do not breathe vapours; keep each {noun} step to no more than {T}.",
    "{code} {H} {ger_cap} {A} must not take longer than {T}.",
]}
_EXPO_HOOD_T = {"*": [
    "{A}: {code} {H} P260/P284: do not breathe vapours; handle it in a fume hood or wear respiratory protection.",
    "{A} ({code}): {H} When {ger} it, work inside the fume hood or wear a respirator.",
    "P284 ({A}): {ger_cap} outside the fume hood is only allowed with respiratory protection.",
]}


def gen_exposure_limit(ctx: GenContext, style: str) -> RuleDraft:
    kind = ctx.weighted({"open_time": 0.5, "duration": 0.25, "hood_or_resp": 0.25})
    pool = ctx.chems_with(volatile=True) if kind != "duration" else ctx.chems_with("toxic", "corrosive")
    chem = ctx.pick(pool)[0]
    code, h = ctx.ghs(chem, INHALATION_CODES)
    sev = GHS_SEVERITY[code]
    A = ctx.name(chem)
    intv = "block" if sev == "high" else "log"
    if kind == "open_time":
        n = float(ctx.choice((30, 60, 90, 120, 180, 300)))
        scope = ctx.choice((("pour", "transfer", "transport"), ("pour", "transfer"), ("open", "pour", "transfer")))
        texts = ctx.render(_EXPO_OPEN_T, style, sev, A=A, H=h, code=code, T=ctx.seconds(n))
        return ctx.draft("exposure_limit", [chem], f"src.open_time <= {fmt(n)}", scope, sev, intv, texts,
                         "expo_open_time")
    if kind == "duration":
        n = float(ctx.choice((30, 60, 120, 300)))
        scope = ctx.choice((("pour", "mix"), ("pour", "transfer", "mix")))
        texts = ctx.render(_EXPO_DUR_T, style, sev, A=A, H=h, code=code, T=ctx.seconds(n), noun=verbs(scope, 2),
                           ger_cap=verbs(scope, 1).capitalize())
        return ctx.draft("exposure_limit", [chem], f"action.duration <= {fmt(n)}", scope, sev, intv, texts,
                         "expo_duration")
    scope = ("open", "pour", "transfer")
    texts = ctx.render(_EXPO_HOOD_T, style, sev, A=A, H=h, code=code, ger=verbs(scope, 1),
                       ger_cap=verbs(scope, 1).capitalize())
    return ctx.draft("exposure_limit", [chem], "in_zone('fume_hood') or has_ppe('respirator')", scope, sev,
                     intv, texts, "expo_hood_or_respirator")


# ----------------------------------------------------------------------------
# 6. temperature_constraint  (expert, lab_manual)
# ----------------------------------------------------------------------------
_TEMP_HEAT_T = {
    "lab_manual": ["{neg} heat {A} above {T}.",
                   "The temperature of {A} must stay at or below {T} during heating.",
                   "When heating {A}, do not exceed {T}."],
    "expert": ["{rneg} set a heating target above {T} for {A}.",
               "Clamp any heat command for {A} to at most {T}.",
               "{rneg} heat {A} to more than {T}."],
}
_TEMP_STRICT_T = {
    "lab_manual": ["{A} must be kept below {T} while it is heated.", "Keep {A} strictly below {T} when heating."],
    "expert": ["Any heating target for {A} must remain below {T}.", "The robot must keep {A} below {T} when heating."],
}
_TEMP_DEVICE_T = {
    "lab_manual": ["{neg} set the hot plate above {T}.", "The hot plate must never be set higher than {T}.",
                   "Hot plate settings above {T} are not allowed."],
    "expert": ["{rneg} command a hot-plate temperature above {T}.", "Clamp every hot-plate setpoint to {T} or less."],
}
_TEMP_COOL_T = {
    "lab_manual": ["Let hot liquids cool to {T} or below before {ger} them.",
                   "{neg} {verb} a liquid that is hotter than {T}.",
                   "Wait until the liquid has cooled to at most {T} before {ger}."],
    "expert": ["{rneg} {verb} a liquid whose temperature exceeds {T}.",
               "Precondition for {ger}: source temperature at or below {T}."],
}
_TEMP_DST_T = {
    "lab_manual": ["{neg} pour {A} into a vessel that is hotter than {T}.",
                   "The receiving vessel must be at {T} or cooler before {A} is added."],
    "expert": ["{rneg} pour {A} into a destination warmer than {T}."],
}
_TEMP_GLASS_T = {
    "lab_manual": ["{neg} pick up glassware hotter than {T}.", "Let glassware cool to {T} before picking it up."],
    "expert": ["{rneg} grasp glassware whose temperature exceeds {T}."],
}


def gen_temperature_constraint(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"high": 0.5, "medium": 0.45, "low": 0.05})
    kind = ctx.weighted({"heat": 0.45, "strict": 0.1, "device": 0.15, "cool": 0.12, "dst": 0.1, "glass": 0.08})
    if kind in ("heat", "strict"):
        chem = ctx.pick([c for c in ctx.reg.chemical_ids if c not in ("Na", "H2O")])[0]
        base = ctx.reg.max_safe_temp(chem) or 60.0
        t = float(ctx.choice(sorted({5 * round(base / 5), 40.0, 50.0, 60.0, 80.0})))
        if kind == "strict":
            texts = ctx.render(_TEMP_STRICT_T, style, sev, A=ctx.name(chem), T=ctx.temp(t))
            return ctx.draft("temperature_constraint", [chem], f"action.target_temperature < {fmt(t)}", ("heat",),
                             sev, "modify", texts, "temp_heat_strict")
        texts = ctx.render(_TEMP_HEAT_T, style, sev, A=ctx.name(chem), T=ctx.temp(t))
        return ctx.draft("temperature_constraint", [chem], f"action.target_temperature <= {fmt(t)}", ("heat",),
                         sev, "modify", texts, "temp_heat")
    if kind == "device":
        t = float(ctx.choice((150, 200, 250)))
        texts = ctx.render(_TEMP_DEVICE_T, style, sev, T=ctx.temp(t))
        return ctx.draft("temperature_constraint", ["hot_plate"], f"action.target_temperature <= {fmt(t)}",
                         ("heat",), sev, "modify", texts, "temp_device")
    if kind == "cool":
        t = float(ctx.choice((40, 50, 60)))
        scope = ctx.choice((("pour", "transfer"), ("pour",)))
        texts = ctx.render(_TEMP_COOL_T, style, sev, T=ctx.temp(t), ger=verbs(scope, 1), verb=verbs(scope, 0))
        return ctx.draft("temperature_constraint", [], f"src.temperature <= {fmt(t)}", scope, sev, "block", texts,
                         "temp_cool_before_pour")
    if kind == "dst":
        chem = ctx.pick(ctx.chems_with("flammable", "peroxide_former"))[0]
        t = float(ctx.choice((30, 35, 40)))
        texts = ctx.render(_TEMP_DST_T, style, sev, A=ctx.name(chem), T=ctx.temp(t))
        return ctx.draft("temperature_constraint", [chem], f"dst.temperature <= {fmt(t)}", ("pour",), sev, "block",
                         texts, "temp_dst")
    t = float(ctx.choice((40, 50, 60)))
    texts = ctx.render(_TEMP_GLASS_T, style, sev, T=ctx.temp(t))
    return ctx.draft("temperature_constraint", ["glassware"], f"src.temperature <= {fmt(t)}", ("pick",), sev, "block",
                     texts, "temp_hot_glass")


# ----------------------------------------------------------------------------
# 7. ventilation_requirement  (pubchem_ghs)
# ----------------------------------------------------------------------------
_VENT_T = {
    "hood": {"*": [
        "{A}: {H} P271: Use only in a well-ventilated area; {ger} must take place inside the fume hood.",
        "{code} {H} P271: Handle {A} only inside a fume hood when {ger}.",
        "{A} ({code}): {verb_cap} it only under the fume hood.",
        "P271 ({A}): {ger} outside the fume hood is prohibited.",
    ]},
    "hood_on": {"*": [
        "{A}: {H} Before {ger} it, make sure the fume hood extraction is switched on.",
        "P271 ({A}): the fume hood must be running whenever you {verb} it.",
        "{code} {H} {verb_cap} {A} only while the fume hood is on.",
    ]},
    "hood_and_on": {"*": [
        "{A}: {H} P271: {ger} must be done inside a fume hood that is switched on.",
        "{code} ({A}): {verb} it only inside a running fume hood.",
    ]},
    "hood_and_sash": {"*": [
        "{A}: {H} P271: {verb} it only inside the fume hood with the sash lowered.",
        "{code} ({A}): {ger} requires the fume hood with its sash pulled down.",
    ]},
}
_VENT_PRED = {
    "hood": "in_zone('fume_hood')",
    "hood_on": "status('fume_hood', 'on')",
    "hood_and_on": "in_zone('fume_hood') and status('fume_hood', 'on')",
    "hood_and_sash": "in_zone('fume_hood') and status('fume_hood', 'sash_lowered')",
}


def gen_ventilation_requirement(ctx: GenContext, style: str) -> RuleDraft:
    kind = ctx.weighted({"hood": 0.55, "hood_on": 0.2, "hood_and_on": 0.15, "hood_and_sash": 0.1})
    pool = ctx.chems_with(volatile=True) + ["NaClO", "H2SO4"]
    k = 2 if (ctx.coin(0.2) and kind == "hood") else 1
    chems = ctx.reg.sort_ids(ctx.pick(pool, k))
    scope = ctx.choice((("open", "pour", "transfer", "mix"), ("open", "pour"), ("heat",), ("pour", "transfer", "mix")))
    code, h = ctx.ghs(chems[0], INHALATION_CODES)
    sev = GHS_SEVERITY[code]
    texts = ctx.render(_VENT_T[kind], style, sev, A=join([ctx.name(c) for c in chems], "and"), H=h, code=code,
                       ger=verbs(scope, 1), verb=verbs(scope, 0), verb_cap=verbs(scope, 0).capitalize())
    # S lists the named chemicals plus entities the predicate queries via status()/distance()
    subs = chems + (["fume_hood"] if "status(" in _VENT_PRED[kind] else [])
    return ctx.draft("ventilation_requirement", subs, _VENT_PRED[kind], scope, sev, "block", texts, f"vent_{kind}")


# ----------------------------------------------------------------------------
# 8. fire_safety  (pubchem_ghs)
# ----------------------------------------------------------------------------
_FIRE_T = {
    "dist": {"*": [
        "{A}: {H} P210: Keep away from heat and hot surfaces; keep it at least {D} from the hot plate.",
        "P210 ({A}): keep containers at a minimum distance of {D} from any hot plate.",
        "{code} {H} Never place {A} closer than {D} to a hot plate.",
    ]},
    "dist_or_off": {"*": [
        "{A}: {H} P210: Do not bring it within {D} of a hot plate that is switched on.",
        "P210 ({A}): keep it at least {D} away from a hot plate unless the hot plate is off.",
    ]},
    "no_heat": {"*": [
        "{A}: {H} Never heat it on a hot plate.",
        "P210 ({A}): heating it on a hot plate is prohibited.",
        "{code} {H} {A} must not be heated directly.",
    ]},
    "flash": {"*": [
        "{A}: {H} Keep it below its flash point; do not heat it above {T}.",
        "P210 ({A}): never warm it beyond {T}.",
        "{code} ({A}): heating above {T} creates a fire risk.",
    ]},
    "hot_zone": {"*": [
        "{A}: {H} P210: Never place it in the heating area.",
        "P210 ({A}): do not set containers of it down near the hot plates.",
        "{code} {H} Keep {A} out of the heating zone.",
    ]},
}


def gen_fire_safety(ctx: GenContext, style: str) -> RuleDraft:
    kind = ctx.weighted({"dist": 0.3, "dist_or_off": 0.2, "no_heat": 0.15, "flash": 0.15, "hot_zone": 0.2})
    pool = ctx.chems_with("flammable")
    chem = ctx.pick(pool)[0]
    code, h = ctx.ghs(chem, ("H224", "H225", "H226", "H260"))
    sev = GHS_SEVERITY[code]
    fields: dict[str, Any] = dict(A=ctx.name(chem), H=h, code=code)
    intv = "block"
    if kind in ("dist", "dist_or_off"):
        d = float(ctx.choice((0.2, 0.3, 0.5)))
        fields["D"] = ctx.dist(d)
        pred = f"distance('hot_plate') >= {fmt(d)}"
        if kind == "dist_or_off":
            pred += " or not status('hot_plate', 'on')"
        scope = ("place", "store", "transport")
        subs = [chem, "hot_plate"]
    elif kind == "no_heat":
        pred, scope, subs = f"not contains(src, '{chem}')", ("heat",), [chem]
    elif kind == "flash":
        t = float(ctx.choice((20, 30, 40)))
        fields["T"] = ctx.temp(t)
        pred, scope, subs, intv = f"action.target_temperature <= {fmt(t)}", ("heat",), [chem], "modify"
    else:
        pred, scope, subs, intv = "not in_zone('hot_zone')", ("place", "store"), [chem], "modify"
    texts = ctx.render(_FIRE_T[kind], style, sev, **fields)
    return ctx.draft("fire_safety", subs, pred, scope, sev, intv, texts, f"fire_{kind}")


# ----------------------------------------------------------------------------
# 9. ppe_requirement  (pubchem_ghs)
# ----------------------------------------------------------------------------
_PPE_SETS = (("gloves",), ("goggles",), ("gloves", "goggles"), ("face_shield",), ("lab_coat",),
             ("gloves", "goggles", "lab_coat"), ("respirator",))
_PPE_T = {"*": [
    "{A}: {H} P280: Wear {ppe}.",
    "{code} {H} When {ger} {A}, wear {ppe}.",
    "P280 ({A}): {ppe} must be worn while {ger} it.",
    "{A} ({code}): {pos} wear {ppe} when you {verb} it.",
]}


def gen_ppe_requirement(ctx: GenContext, style: str) -> RuleDraft:
    items = ctx.choice(_PPE_SETS)
    pool = ctx.chems_with("corrosive", "toxic") if items != ("respirator",) else ctx.chems_with(volatile=True)
    k = int(ctx.choice((1, 1, 1, 2, 3)))
    chems = ctx.reg.sort_ids(ctx.pick(pool, k))
    scope = ctx.choice((("pick", "pour", "transfer", "mix"), ("pour", "transfer"),
                        ("pick", "pour", "transfer", "transport", "mix"), ("open", "pour")))
    code, h = ctx.ghs(chems[0], ("H314", "H318", "H331"))
    sev = GHS_SEVERITY[code]
    pred = " and ".join(f"has_ppe('{x}')" for x in items)
    texts = ctx.render(_PPE_T, style, sev, A=join([ctx.name(c) for c in chems], "or"), H=h, code=code,
                       ppe=join([PPE_TEXT[x] for x in items], "and"), ger=verbs(scope, 1), verb=verbs(scope, 0))
    return ctx.draft("ppe_requirement", chems, pred, scope, sev, "block", texts, "ppe")


# ----------------------------------------------------------------------------
# 10. velocity_limit  (lab_manual)
# ----------------------------------------------------------------------------
_VEL_T = {
    "carry": {"*": [
        "{pos} {verb} {obj} slowly, at no more than {V}.",
        "{neg} move faster than {V} while {ger} {obj}.",
        "When {ger} {obj}, keep the speed at or below {V}.",
        "The speed limit for {ger} {obj} is {V}.",
    ]},
    "pour_still": {"*": [
        "Keep the arm almost still while pouring; its speed must stay at or below {V}.",
        "{neg} move the robot faster than {V} during a pour.",
    ]},
    "full_slow": {"*": [
        "Containers that are more than half full must be carried at no more than {V}.",
        "{neg} carry a container filled above half its capacity faster than {V}.",
    ]},
}


def gen_velocity_limit(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"medium": 0.7, "low": 0.15, "high": 0.15})
    kind = ctx.weighted({"carry": 0.7, "pour_still": 0.15, "full_slow": 0.15})
    if kind == "pour_still":
        v = float(ctx.choice((0.05, 0.1)))
        texts = ctx.render(_VEL_T[kind], style, sev, V=ctx.speed(v))
        return ctx.draft("velocity_limit", [], f"robot.speed <= {fmt(v)}", ("pour",), sev, "modify", texts,
                         "vel_pour_still")
    if kind == "full_slow":
        v = float(ctx.choice((0.1, 0.15)))
        texts = ctx.render(_VEL_T[kind], style, sev, V=ctx.speed(v))
        return ctx.draft("velocity_limit", ["glassware"], f"src.fill_ratio <= 0.5 or action.speed <= {fmt(v)}",
                         ("transport",), sev, "modify", texts, "vel_full_slow")
    v = float(ctx.choice((0.1, 0.15, 0.2, 0.25, 0.3, 0.5)))
    scope = ctx.choice((("transport",), ("transport", "move"), ("pick", "place", "transport")))
    if ctx.coin(0.3 if ctx.require is None else 1.0):
        chem = ctx.pick(ctx.chems_with("corrosive", "toxic", "flammable"))[0]
        subs, obj = [chem], f"containers of {ctx.name(chem)}"
    else:
        subs, obj = ["glassware"], ctx.choice(("glassware", "glass containers", "beakers"))
    texts = ctx.render(_VEL_T["carry"], style, sev, V=ctx.speed(v), obj=obj, ger=verbs(scope, 1),
                       verb=verbs(scope, 0))
    return ctx.draft("velocity_limit", subs, f"action.speed <= {fmt(v)}", scope, sev, "modify", texts, "vel_carry")


# ----------------------------------------------------------------------------
# 11. waste_handling  (pubchem_ghs, expert, lab_manual)
# ----------------------------------------------------------------------------
_WASTE_LABEL_T = {
    "lab_manual": ["{pos} dispose of {A} only in the {stream} container.",
                   "{A} waste belongs in the {stream} container.",
                   "{neg} pour {A} waste into anything but the {stream} container."],
    "expert": ["{rpos} route {A} waste to the {stream} container.",
               "{rneg} dispose of {A} in any container other than the {stream} container."],
    "pubchem_ghs": ["{A}: P501 Dispose of contents into the {stream} container.",
                    "P501 ({A}): dispose of contents and container as {stream}.",
                    "{A} ({code}): waste must go to the {stream} container only."],
}
_WASTE_INC_T = {"*": [
    "{neg} add waste to a waste container that holds incompatible chemicals.",
    "Before disposal, make sure the waste is compatible with what is already in the waste container.",
    "{rneg} dispose of waste into a container whose contents are incompatible with it.",
]}
_WASTE_STATUS_T = {
    "labeled": {"*": ["{neg} dispose of waste into an unlabeled waste container.",
                      "Waste may only go into a labeled waste container.",
                      "{pos} check that the waste container is labeled before disposal."]},
    "not_full": {"*": ["{neg} add waste to a full waste container.",
                       "Check that the waste container is not full before disposing of waste.",
                       "{rneg} dispose of waste into a waste container that is already full."]},
}


def gen_waste_handling(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"medium": 0.6, "high": 0.25, "low": 0.15})
    kind = ctx.weighted({"label": 0.6, "incompatible": 0.15, "labeled": 0.125, "not_full": 0.125})
    if kind == "label":
        pool = [c for c in ctx.reg.chemical_ids if ctx.reg.waste_stream(c) and c != "H2O"]
        chem = ctx.pick(pool)[0]
        stream = ctx.reg.waste_stream(chem)
        code, _ = ctx.ghs(chem)
        texts = ctx.render(_WASTE_LABEL_T, style, sev, A=ctx.name(chem), stream=WASTE_TEXT[stream], code=code)
        return ctx.draft("waste_handling", [chem], f"dst.label == '{stream}'", ("dispose",), sev, "modify", texts,
                         "waste_label")
    if kind == "incompatible":
        texts = ctx.render(_WASTE_INC_T, style, sev)
        return ctx.draft("waste_handling", ["waste_container"], "not incompatible(src, dst)", ("dispose",), sev,
                         "block", texts, "waste_incompatible")
    texts = ctx.render(_WASTE_STATUS_T[kind], style, sev)
    return ctx.draft("waste_handling", ["waste_container"], f"status('waste_container', '{kind}')", ("dispose",),
                     sev, "block", texts, f"waste_{kind}")


# ----------------------------------------------------------------------------
# 12. spill_prevention  (lab_manual)
# ----------------------------------------------------------------------------
_SPILL_T = {
    "rate": {"*": [
        "{pos} pour {obj} slowly, at no more than {R}, to avoid splashing.",
        "{neg} pour {obj} faster than {R}.",
        "The pouring rate for {obj} must not exceed {R}.",
    ]},
    "tilt": {"*": [
        "{neg} tilt the container more than {Ang} while pouring.",
        "Keep the pouring angle at or below {Ang} to avoid spills.",
        "When pouring, limit the tilt of the vessel to {Ang}.",
    ]},
    "open_slow": {"*": [
        "Open containers must be carried at no more than {V} to prevent spills.",
        "{neg} carry an uncapped container faster than {V}.",
    ]},
}


def gen_spill_prevention(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"medium": 0.7, "low": 0.2, "high": 0.1})
    kind = ctx.weighted({"rate": 0.55, "tilt": 0.25, "open_slow": 0.2})
    if kind == "rate":
        r = float(ctx.choice((10, 15, 20, 25, 30)))
        scope = ctx.choice((("pour", "transfer"), ("pour",)))
        subs: list[str] = []
        obj = "liquids"
        if ctx.coin(0.35 if ctx.require is None else 1.0):
            chem = ctx.pick(ctx.chems_with("corrosive"))[0]
            subs, obj = [chem], ctx.name(chem)
        texts = ctx.render(_SPILL_T[kind], style, sev, R=ctx.rate(r), obj=obj)
        return ctx.draft("spill_prevention", subs, f"action.pour_rate <= {fmt(r)}", scope, sev, "modify", texts,
                         "spill_rate")
    if kind == "tilt":
        a = float(ctx.choice((60, 75, 90)))
        texts = ctx.render(_SPILL_T[kind], style, sev, Ang=ctx.choice((f"{fmt(a)} degrees", f"{fmt(a)}°")))
        return ctx.draft("spill_prevention", [], f"action.tilt_angle <= {fmt(a)}", ("pour",), sev, "modify", texts,
                         "spill_tilt")
    v = float(ctx.choice((0.1, 0.15)))
    texts = ctx.render(_SPILL_T[kind], style, sev, V=ctx.speed(v))
    return ctx.draft("spill_prevention", [], f"not src.lid_open or action.speed <= {fmt(v)}", ("transport",), sev,
                     "modify", texts, "spill_open_slow")


# ----------------------------------------------------------------------------
# 13. container_capacity  (lab_manual)
# ----------------------------------------------------------------------------
_CAP_T = {
    "fill": {"*": [
        "{neg} fill a beaker or flask above {F} of its capacity.",
        "Containers must not be filled beyond {F} of their volume.",
        "When {ger}, leave headspace: the receiving vessel may be filled to at most {F}.",
    ]},
    "heat": {"*": ["{neg} heat a flask that is more than {F} full.",
                   "Only heat vessels that are filled to {F} or less."]},
    "shake": {"*": ["{neg} shake a container that is more than {F} full.",
                    "Containers filled above {F} must not be shaken."]},
    "volume": {"*": ["{neg} dispense more than {Vol} in a single operation.",
                     "Each pour or dispense step is limited to {Vol}."]},
}


def gen_container_capacity(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"medium": 0.7, "low": 0.15, "high": 0.15})
    kind = ctx.weighted({"fill": 0.5, "heat": 0.2, "shake": 0.15, "volume": 0.15})
    if kind == "fill":
        f = float(ctx.choice((0.5, 0.75, 0.8, 0.9)))
        scope = ctx.choice((("pour", "transfer", "dispense"), ("pour", "transfer")))
        texts = ctx.render(_CAP_T[kind], style, sev, F=ctx.fraction(f), ger=verbs(scope, 1))
        subs = ["glassware"]
        return ctx.draft("container_capacity", subs, f"action.fill_after <= {fmt(f)}", scope, sev, "modify", texts,
                         "cap_fill")
    if kind in ("heat", "shake"):
        f = float(ctx.choice((0.5, 0.75)) if kind == "heat" else ctx.choice((0.75, 0.8)))
        texts = ctx.render(_CAP_T[kind], style, sev, F=ctx.fraction(f))
        return ctx.draft("container_capacity", ["glassware"], f"src.fill_ratio <= {fmt(f)}", (kind,), sev, "block",
                         texts, f"cap_{kind}")
    v = float(ctx.choice((50, 100, 200, 500)))
    texts = ctx.render(_CAP_T[kind], style, sev, Vol=ctx.volume(v))
    return ctx.draft("container_capacity", [], f"action.volume <= {fmt(v)}", ("pour", "dispense"), sev, "modify",
                     texts, "cap_volume")


# ----------------------------------------------------------------------------
# 14. workspace_zone  (lab_manual)
# ----------------------------------------------------------------------------
_WS_T = {
    "workspace": {"*": [
        "{pos} keep the end effector inside the marked robot workspace.",
        "{neg} move the gripper outside the designated workspace.",
        "While {ger}, the arm must stay within the robot workspace.",
    ]},
    "hot_zone": {"*": ["{neg} move the arm through the heating area.",
                       "Keep the gripper out of the hot-plate zone while {ger}."]},
    "dist": {"*": ["Keep the gripper at least {D} away from the hot plate while {ger}.",
                   "{neg} bring the end effector closer than {D} to the hot plate."]},
    "sink": {"*": ["{neg} carry {A} over the sink area.", "Keep {A} away from the sink when carrying it."]},
}


def gen_workspace_zone(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"high": 0.6, "medium": 0.4})
    kind = ctx.weighted({"workspace": 0.45, "hot_zone": 0.2, "dist": 0.2, "sink": 0.15})
    scope = ctx.choice((("move", "transport", "pick", "place"), ("move", "transport")))
    if kind == "workspace":
        texts = ctx.render(_WS_T[kind], style, sev, ger=verbs(scope, 1))
        return ctx.draft("workspace_zone", [], "in_zone('workspace')", scope, sev, "modify", texts, "ws_workspace")
    if kind == "hot_zone":
        texts = ctx.render(_WS_T[kind], style, sev, ger=verbs(scope, 1))
        return ctx.draft("workspace_zone", [], "not in_zone('hot_zone')", scope, sev, "block", texts, "ws_hot_zone")
    if kind == "dist":
        d = float(ctx.choice((0.1, 0.15, 0.2)))
        texts = ctx.render(_WS_T[kind], style, sev, D=ctx.dist(d), ger=verbs(scope, 1))
        return ctx.draft("workspace_zone", ["hot_plate"], f"distance('hot_plate') >= {fmt(d)}", scope, sev, "block",
                         texts, "ws_dist")
    chem = ctx.pick(ctx.chems_with("water_reactive"))[0]
    texts = ctx.render(_WS_T[kind], style, sev, A=ctx.name(chem))
    return ctx.draft("workspace_zone", [chem], "not in_zone('sink')", ("transport",), sev, "block", texts, "ws_sink")


# ----------------------------------------------------------------------------
# 15. force_limit  (lab_manual)
# ----------------------------------------------------------------------------
_FORCE_T = {
    "grip": {"*": [
        "{neg} grip {obj} with more than {N}.",
        "Limit the gripping force on {obj} to {N}.",
        "When {ger} {obj}, the grip force must stay at or below {N}.",
    ]},
    "shake": {"*": ["While shaking, the gripper force must stay at or below {N}.",
                    "{neg} squeeze a container harder than {N} while shaking it."]},
    "range": {"*": ["Grip {obj} firmly but gently, with a force between {lo} N and {hi} N.",
                    "The grip force on {obj} must be at least {lo} N and at most {hi} N."]},
}


def gen_force_limit(ctx: GenContext, style: str) -> RuleDraft:
    sev = _sev(ctx, {"medium": 0.7, "high": 0.2, "low": 0.1})
    kind = ctx.weighted({"grip": 0.65, "shake": 0.15, "range": 0.2})
    obj = ctx.choice(("glassware", "glass containers", "beakers and flasks"))
    if kind == "shake":
        f = float(ctx.choice((20, 25)))
        texts = ctx.render(_FORCE_T[kind], style, sev, N=f"{fmt(f)} N")
        return ctx.draft("force_limit", [], f"robot.gripper_force <= {fmt(f)}", ("shake",), sev, "modify", texts,
                         "force_shake")
    scope = ctx.choice((("pick", "transport"), ("pick", "transport", "pour", "shake"), ("pick",)))
    if kind == "range":
        lo, hi = ctx.choice(((5, 20), (5, 25), (8, 30)))
        texts = ctx.render(_FORCE_T[kind], style, sev, obj=obj, lo=lo, hi=hi)
        return ctx.draft("force_limit", ["glassware"], f"action.grip_force >= {lo} and action.grip_force <= {hi}",
                         scope, sev, "modify", texts, "force_range")
    f = float(ctx.choice((10, 15, 20, 25, 30)))
    texts = ctx.render(_FORCE_T[kind], style, sev, obj=obj, N=f"{fmt(f)} N", ger=verbs(scope, 1))
    return ctx.draft("force_limit", ["glassware"], f"action.grip_force <= {fmt(f)}", scope, sev, "modify", texts,
                     "force_grip")


GENERATORS: dict[str, Callable[[GenContext, str], RuleDraft]] = {
    "chemical_incompatibility": gen_chemical_incompatibility,
    "equipment_check": gen_equipment_check,
    "sequence_constraint": gen_sequence_constraint,
    "exposure_limit": gen_exposure_limit,
    "temperature_constraint": gen_temperature_constraint,
    "ventilation_requirement": gen_ventilation_requirement,
    "fire_safety": gen_fire_safety,
    "ppe_requirement": gen_ppe_requirement,
    "velocity_limit": gen_velocity_limit,
    "waste_handling": gen_waste_handling,
    "spill_prevention": gen_spill_prevention,
    "container_capacity": gen_container_capacity,
    "storage_constraint": gen_storage_constraint,
    "workspace_zone": gen_workspace_zone,
    "force_limit": gen_force_limit,
}


def generate_rule(ctype: str, style: str, rng: np.random.Generator, registry: Registry | None = None,
                  require: str | None = None, exclude: Iterable[str] = ()) -> RuleDraft:
    ctx = GenContext(rng, registry, require=require, exclude=exclude)
    return GENERATORS[ctype](ctx, style)
