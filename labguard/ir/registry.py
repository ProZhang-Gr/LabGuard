"""Runtime substance/entity registry (Sec. 3.4, 4.1, Appendix B.1).

The registry serves three purposes:
  1. resolve surface mentions ("muriatic acid", "bleach") to canonical ids
     ("HCl", "NaClO") -- used by schema validation and by the hybrid grounder;
  2. store the incompatibility table inc(.) used by compat(c_a, c_b) (Eq. 7);
  3. store hazard metadata used by monitors, the risk scorer and the
     simulator's outcome oracle.

The 21 canonical substances/entities are our reconstruction (16 chemicals +
5 laboratory entities); the paper only reports the count.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable


class RegistryError(ValueError):
    """Raised when a substance/entity mention cannot be resolved."""


@dataclass(frozen=True)
class SubstanceInfo:
    id: str
    name: str
    kind: str  # "chemical" | "entity"
    aliases: tuple[str, ...] = ()
    hazards: frozenset[str] = frozenset()
    max_safe_temp: float | None = None  # degC; above this the oracle flags overheating
    volatile: bool = False
    waste_stream: str | None = None


# GHS-like hazard classes and their contribution to the chemical risk term rho_c
HAZARD_WEIGHTS: dict[str, float] = {
    "water_reactive": 0.9,
    "oxidizer": 0.7,
    "toxic": 0.7,
    "corrosive": 0.6,
    "flammable": 0.6,
    "peroxide_former": 0.5,
    "acid": 0.3,
    "base": 0.3,
    "halogenated": 0.3,
}

_CHEMICALS: tuple[SubstanceInfo, ...] = (
    SubstanceInfo("HCl", "hydrochloric acid", "chemical",
                  ("hydrochloric acid", "muriatic acid", "hydrogen chloride", "HCl"),
                  frozenset({"corrosive", "acid", "toxic"}), 60.0, True, "acid_waste"),
    SubstanceInfo("NaClO", "sodium hypochlorite", "chemical",
                  ("sodium hypochlorite", "bleach", "hypochlorite", "NaClO", "NaOCl"),
                  frozenset({"oxidizer", "corrosive"}), 40.0, False, "oxidizer_waste"),
    SubstanceInfo("H2SO4", "sulfuric acid", "chemical",
                  ("sulfuric acid", "sulphuric acid", "oil of vitriol", "H2SO4"),
                  frozenset({"corrosive", "acid"}), 150.0, False, "acid_waste"),
    SubstanceInfo("HNO3", "nitric acid", "chemical",
                  ("nitric acid", "HNO3"),
                  frozenset({"corrosive", "acid", "oxidizer"}), 80.0, True, "acid_waste"),
    SubstanceInfo("NaOH", "sodium hydroxide", "chemical",
                  ("sodium hydroxide", "caustic soda", "lye", "NaOH"),
                  frozenset({"corrosive", "base"}), 90.0, False, "base_waste"),
    SubstanceInfo("NH3", "ammonia", "chemical",
                  ("ammonia", "ammonium hydroxide", "aqueous ammonia", "NH3", "NH4OH"),
                  frozenset({"toxic", "corrosive", "base"}), 30.0, True, "base_waste"),
    SubstanceInfo("H2O2", "hydrogen peroxide", "chemical",
                  ("hydrogen peroxide", "peroxide", "H2O2"),
                  frozenset({"oxidizer", "corrosive"}), 50.0, False, "oxidizer_waste"),
    SubstanceInfo("KMnO4", "potassium permanganate", "chemical",
                  ("potassium permanganate", "permanganate", "KMnO4"),
                  frozenset({"oxidizer"}), 150.0, False, "oxidizer_waste"),
    SubstanceInfo("Na", "sodium metal", "chemical",
                  ("sodium metal", "metallic sodium", "elemental sodium"),
                  frozenset({"water_reactive", "flammable", "corrosive"}), 90.0, False, None),
    SubstanceInfo("C2H5OH", "ethanol", "chemical",
                  ("ethanol", "ethyl alcohol", "EtOH", "C2H5OH", "alcohol"),
                  frozenset({"flammable"}), 60.0, True, "organic_waste"),
    SubstanceInfo("CH3OH", "methanol", "chemical",
                  ("methanol", "methyl alcohol", "MeOH", "CH3OH", "wood alcohol"),
                  frozenset({"flammable", "toxic"}), 55.0, True, "organic_waste"),
    SubstanceInfo("C3H6O", "acetone", "chemical",
                  ("acetone", "propanone", "dimethyl ketone", "C3H6O"),
                  frozenset({"flammable"}), 45.0, True, "organic_waste"),
    SubstanceInfo("C4H10O", "diethyl ether", "chemical",
                  ("diethyl ether", "ethyl ether", "ether", "Et2O", "C4H10O"),
                  frozenset({"flammable", "peroxide_former"}), 30.0, True, "organic_waste"),
    SubstanceInfo("CHCl3", "chloroform", "chemical",
                  ("chloroform", "trichloromethane", "CHCl3"),
                  frozenset({"toxic", "halogenated"}), 50.0, True, "halogenated_waste"),
    SubstanceInfo("CH3COOH", "acetic acid", "chemical",
                  ("acetic acid", "glacial acetic acid", "ethanoic acid", "CH3COOH", "AcOH"),
                  frozenset({"corrosive", "acid", "flammable"}), 100.0, True, "acid_waste"),
    SubstanceInfo("H2O", "water", "chemical",
                  ("water", "distilled water", "deionized water", "DI water", "H2O"),
                  frozenset(), 95.0, False, "aqueous_waste"),
)

_ENTITIES: tuple[SubstanceInfo, ...] = (
    SubstanceInfo("hot_plate", "hot plate", "entity",
                  ("hot plate", "hotplate", "heating plate", "heating mantle", "hot_plate")),
    SubstanceInfo("fume_hood", "fume hood", "entity",
                  ("fume hood", "fume cupboard", "fume_hood", "hood")),
    SubstanceInfo("centrifuge", "centrifuge", "entity", ("centrifuge", "centrifuges")),
    SubstanceInfo("glassware", "glassware", "entity",
                  ("glassware", "glass container", "glass containers", "beaker", "beakers",
                   "flask", "flasks", "test tube", "test tubes", "graduated cylinder",
                   "graduated cylinders", "glass vessel", "glass vessels")),
    SubstanceInfo("waste_container", "waste container", "entity",
                  ("waste container", "waste containers", "waste bottle", "waste bottles",
                   "waste jug", "waste receptacle", "waste_container")),
)

# Symmetric incompatibility table: inc(c) in Eq. 7.
_INCOMPATIBLE_PAIRS: tuple[tuple[str, str], ...] = (
    ("HCl", "NaClO"), ("HCl", "KMnO4"), ("HCl", "NaOH"), ("HCl", "NH3"), ("HCl", "Na"),
    ("HCl", "HNO3"),
    ("NaClO", "NH3"), ("NaClO", "H2SO4"), ("NaClO", "HNO3"), ("NaClO", "CH3COOH"),
    ("NaClO", "C2H5OH"), ("NaClO", "C3H6O"), ("NaClO", "CH3OH"), ("NaClO", "H2O2"),
    ("H2SO4", "NaOH"), ("H2SO4", "KMnO4"), ("H2SO4", "Na"), ("H2SO4", "NH3"),
    ("HNO3", "C2H5OH"), ("HNO3", "CH3OH"), ("HNO3", "C3H6O"), ("HNO3", "CH3COOH"),
    ("HNO3", "NaOH"), ("HNO3", "NH3"), ("HNO3", "Na"), ("HNO3", "C4H10O"),
    ("NaOH", "CH3COOH"), ("NaOH", "CHCl3"),
    ("H2O2", "KMnO4"), ("H2O2", "C3H6O"), ("H2O2", "Na"),
    ("KMnO4", "C2H5OH"), ("KMnO4", "CH3OH"), ("KMnO4", "C3H6O"), ("KMnO4", "C4H10O"),
    ("Na", "H2O"), ("Na", "C2H5OH"), ("Na", "CH3OH"), ("Na", "CHCl3"), ("Na", "CH3COOH"),
)

# Named spatial zones (robot base frame, metres).  Monitors only see names;
# the boxes are part of the runtime state and can be overridden per scene.
DEFAULT_ZONES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "workspace": ((0.20, -0.42, 0.00), (0.72, 0.42, 0.60)),
    "fume_hood": ((0.52, -0.42, 0.00), (0.72, -0.20, 0.60)),
    "hot_zone": ((0.40, 0.18, 0.00), (0.60, 0.38, 0.30)),
    "acid_cabinet": ((0.22, 0.30, 0.00), (0.34, 0.42, 0.30)),
    "base_cabinet": ((0.22, -0.42, 0.00), (0.34, -0.30, 0.30)),
    "flammable_cabinet": ((0.36, -0.42, 0.00), (0.48, -0.30, 0.30)),
    "oxidizer_shelf": ((0.36, 0.30, 0.00), (0.48, 0.42, 0.30)),
    "general_shelf": ((0.22, -0.10, 0.00), (0.30, 0.10, 0.30)),
    "waste_area": ((0.60, 0.20, 0.00), (0.72, 0.42, 0.30)),
    "sink": ((0.60, -0.15, 0.00), (0.72, 0.05, 0.30)),
}

PPE_ITEMS: tuple[str, ...] = ("gloves", "goggles", "lab_coat", "face_shield", "respirator")
WASTE_STREAMS: tuple[str, ...] = (
    "halogenated_waste", "organic_waste", "aqueous_waste", "acid_waste", "base_waste",
    "oxidizer_waste",
)
# Flags accepted by status(entity, flag)
EQUIPMENT_FLAGS: dict[str, tuple[str, ...]] = {
    "hot_plate": ("on", "calibrated", "clean"),
    "fume_hood": ("on", "sash_lowered"),
    "centrifuge": ("balanced", "lid_closed", "on"),
    "glassware": ("intact", "clean", "dry"),
    "waste_container": ("labeled", "closed", "not_full"),
}
# Step names accepted by done(step): task tags plus a few procedural steps
EXTRA_STEPS: tuple[str, ...] = ("balance", "calibrate", "label", "rinse", "vent", "preheat", "neutralize")


def _norm(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_QUALIFIERS = re.compile(
    r"\b(concentrated|conc|dilute|diluted|aqueous|solution|solutions|pure|anhydrous|fuming|"
    r"commercial|household|any|all|the|a|an)\b"
)


@dataclass
class Registry:
    substances: dict[str, SubstanceInfo]
    incompatible_pairs: set[frozenset[str]] = field(default_factory=set)
    zones: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = field(
        default_factory=lambda: dict(DEFAULT_ZONES))
    ppe_items: tuple[str, ...] = PPE_ITEMS
    waste_streams: tuple[str, ...] = WASTE_STREAMS
    equipment_flags: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(EQUIPMENT_FLAGS))
    extra_steps: tuple[str, ...] = EXTRA_STEPS

    def __post_init__(self) -> None:
        self._alias: dict[str, str] = {}
        self._case_sensitive: dict[str, str] = {}
        for sid, info in self.substances.items():
            for alias in (sid, info.name, *info.aliases):
                # Short formula-like aliases ("Na", "HCl") are matched case-sensitively
                # in free text to avoid false hits on ordinary words.
                if len(alias) <= 4 and any(ch.isupper() for ch in alias):
                    self._case_sensitive[alias] = sid
                self._alias.setdefault(_norm(alias), sid)
        self._inc: dict[str, set[str]] = {sid: set() for sid in self.substances}
        for pair in self.incompatible_pairs:
            a, b = tuple(pair)
            self._inc[a].add(b)
            self._inc[b].add(a)
        # Alias patterns for mention scanning, longest first.
        self._scan: list[tuple[re.Pattern[str], str]] = []
        case_sensitive_keys = {_norm(a) for a in self._case_sensitive}
        for alias, sid in sorted(self._alias.items(), key=lambda kv: -len(kv[0])):
            if alias in case_sensitive_keys:
                continue
            self._scan.append((re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"), sid))
        for alias, sid in sorted(self._case_sensitive.items(), key=lambda kv: -len(kv[0])):
            self._scan.append((re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"), sid))

    # ------------------------------------------------------------------ build
    @classmethod
    def default(cls) -> "Registry":
        subs = {s.id: s for s in (*_CHEMICALS, *_ENTITIES)}
        pairs = {frozenset(p) for p in _INCOMPATIBLE_PAIRS}
        return cls(substances=subs, incompatible_pairs=pairs)

    # ---------------------------------------------------------------- queries
    @property
    def ids(self) -> list[str]:
        return list(self.substances)

    @property
    def chemical_ids(self) -> list[str]:
        return [s for s, i in self.substances.items() if i.kind == "chemical"]

    @property
    def entity_ids(self) -> list[str]:
        return [s for s, i in self.substances.items() if i.kind == "entity"]

    def is_chemical(self, sid: str) -> bool:
        info = self.substances.get(sid)
        return info is not None and info.kind == "chemical"

    def is_entity(self, sid: str) -> bool:
        info = self.substances.get(sid)
        return info is not None and info.kind == "entity"

    def order_key(self, sid: str) -> int:
        try:
            return self.ids.index(sid)
        except ValueError:
            return len(self.substances)

    def sort_ids(self, ids: Iterable[str]) -> list[str]:
        return sorted(set(ids), key=self.order_key)

    def resolve(self, mention: str, fuzzy: bool = True) -> str | None:
        """Map a surface mention to a canonical id (None if unresolvable)."""
        if mention in self.substances:
            return mention
        if mention in self._case_sensitive:
            return self._case_sensitive[mention]
        key = _norm(mention)
        if key in self._alias:
            return self._alias[key]
        stripped = _norm(_QUALIFIERS.sub(" ", key))
        if stripped in self._alias:
            return self._alias[stripped]
        if fuzzy and stripped:
            match = difflib.get_close_matches(stripped, list(self._alias), n=1, cutoff=0.88)
            if match:
                return self._alias[match[0]]
        return None

    def resolve_strict(self, mention: str) -> str:
        sid = self.resolve(mention)
        if sid is None:
            raise RegistryError(f"unresolvable substance/entity mention: {mention!r}")
        return sid

    def find_mentions(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Scan free text for registry mentions (longest match wins, no overlaps)."""
        taken: list[tuple[int, int]] = []
        found: list[tuple[str, tuple[int, int]]] = []
        lowered = text.lower()
        for pattern, sid in self._scan:
            haystack = text if pattern.pattern.startswith("(?<![A-Za-z0-9])") else lowered
            for m in pattern.finditer(haystack):
                span = m.span()
                if any(not (span[1] <= a or span[0] >= b) for a, b in taken):
                    continue
                taken.append(span)
                found.append((sid, span))
        found.sort(key=lambda x: x[1][0])
        return found

    def inc(self, sid: str) -> set[str]:
        return self._inc.get(sid, set())

    def compat(self, a: str, b: str) -> int:
        """Eq. 7: 1 iff neither substance lists the other as incompatible."""
        return int(b not in self.inc(a) and a not in self.inc(b))

    def hazards(self, sid: str) -> frozenset[str]:
        info = self.substances.get(sid)
        return info.hazards if info else frozenset()

    def has_hazard(self, sid: str, hazard_class: str) -> bool:
        return hazard_class in self.hazards(sid)

    def hazard_score(self, sid: str) -> float:
        hz = self.hazards(sid)
        return max((HAZARD_WEIGHTS.get(h, 0.2) for h in hz), default=0.0)

    def max_safe_temp(self, sid: str) -> float | None:
        info = self.substances.get(sid)
        return info.max_safe_temp if info else None

    def waste_stream(self, sid: str) -> str | None:
        info = self.substances.get(sid)
        return info.waste_stream if info else None

    def surface_names(self, sid: str) -> list[str]:
        info = self.substances[sid]
        names = [info.name, *info.aliases]
        seen: list[str] = []
        for n in names:
            if n not in seen:
                seen.append(n)
        return seen

    def step_names(self) -> set[str]:
        from labguard.vocab import TASK_TAGS

        return set(TASK_TAGS) | set(self.extra_steps)


_DEFAULT: Registry | None = None


def default_registry() -> Registry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Registry.default()
    return _DEFAULT
