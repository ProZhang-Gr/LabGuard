"""Closed vocabularies used by LabGuard-IR (Sec. 3.2, Fig. 1c, Table 7).

The paper reports 15 constraint types (grouped into four families), 21
substances/entities, 21 task tags and 3 severity levels.  The concrete
task-tag and substance lists are not published, so the ones below are our
reconstruction aligned with LabUtopia tasks (pick/place/pour/heat/...).
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Constraint types t (Table 7 order) and their four families (Sec. 3.3)
# --------------------------------------------------------------------------
CONSTRAINT_TYPES: tuple[str, ...] = (
    "chemical_incompatibility",
    "equipment_check",
    "sequence_constraint",
    "exposure_limit",
    "temperature_constraint",
    "ventilation_requirement",
    "fire_safety",
    "ppe_requirement",
    "velocity_limit",
    "waste_handling",
    "spill_prevention",
    "container_capacity",
    "storage_constraint",
    "workspace_zone",
    "force_limit",
)

CONSTRAINT_FAMILIES: dict[str, tuple[str, ...]] = {
    "material_compatibility": ("chemical_incompatibility", "storage_constraint"),
    "physical_quantitative": (
        "temperature_constraint",
        "container_capacity",
        "velocity_limit",
        "force_limit",
        "workspace_zone",
    ),
    "process_equipment": ("equipment_check", "sequence_constraint", "spill_prevention"),
    "environment_ppe_waste": (
        "exposure_limit",
        "ventilation_requirement",
        "fire_safety",
        "ppe_requirement",
        "waste_handling",
    ),
}
TYPE_TO_FAMILY: dict[str, str] = {t: fam for fam, ts in CONSTRAINT_FAMILIES.items() for t in ts}

# Per-type counts of the 812-example corpus (Table 7).  Dividing by 4
# (seed + 3 paraphrases) gives the 203 seed rules.
PAPER_TYPE_COUNTS: dict[str, int] = {
    "chemical_incompatibility": 132,
    "equipment_check": 100,
    "sequence_constraint": 80,
    "exposure_limit": 68,
    "temperature_constraint": 64,
    "ventilation_requirement": 60,
    "fire_safety": 56,
    "ppe_requirement": 52,
    "velocity_limit": 40,
    "waste_handling": 36,
    "spill_prevention": 36,
    "container_capacity": 24,
    "storage_constraint": 24,
    "workspace_zone": 20,
    "force_limit": 20,
}

# --------------------------------------------------------------------------
# Action scope T: 21 robot-task tags
# --------------------------------------------------------------------------
TASK_TAGS: tuple[str, ...] = (
    "pick", "place", "press", "shake", "stir", "open", "close",
    "pour", "transfer", "mix", "heat", "cool", "transport", "clean",
    "dispense", "weigh", "centrifuge", "store", "dispose", "move", "inspect",
)

# Tags whose semantics move material from `src` into `dst`.
TRANSFER_TAGS = frozenset({"pour", "transfer", "mix", "dispense", "dispose"})
# Tags whose `dst` is a location/zone rather than a container.
PLACEMENT_TAGS = frozenset({"place", "store"})
# Tags that move the end effector (kinematic monitors are most relevant here).
MOTION_TAGS = frozenset({"move", "transport", "pick", "place", "store"})

# --------------------------------------------------------------------------
# Severity v, intervention mode i, runtime verdict v_t
# --------------------------------------------------------------------------
SEVERITIES: tuple[str, ...] = ("low", "medium", "high")
SEVERITY_RANK: dict[str, int] = {s: k for k, s in enumerate(SEVERITIES)}

# Intervention mode stored in the IR: what to do once a violation is found.
INTERVENTIONS: tuple[str, ...] = ("log", "modify", "block")

# Runtime verdict v_t in {pass, modify, block} (Eq. 1).
VERDICTS: tuple[str, ...] = ("pass", "modify", "block")
VERDICT_RANK: dict[str, int] = {v: k for k, v in enumerate(VERDICTS)}

# --------------------------------------------------------------------------
# LabGuard-Bench metadata (Sec. 3.3, Appendix D)
# --------------------------------------------------------------------------
SOURCE_FAMILIES: tuple[str, ...] = ("lab_manual", "pubchem_ghs", "expert", "chem_safety_ref")
PAPER_SOURCE_COUNTS: dict[str, int] = {
    "lab_manual": 71,
    "pubchem_ghs": 62,
    "expert": 50,
    "chem_safety_ref": 20,
}
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
PAPER_DIFFICULTY_COUNTS: dict[str, int] = {"easy": 71, "medium": 105, "hard": 27}
N_SEED_RULES = 203
N_PARAPHRASES_PER_SEED = 3  # 203 * (1 + 3) = 812 labeled examples
