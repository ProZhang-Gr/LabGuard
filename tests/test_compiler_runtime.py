"""Compiler (Sec. 4.3) and runtime safety layer (Sec. 4.4)."""

import numpy as np
import pytest

from labguard.compiler.compiler import Compiler
from labguard.core.action import Action
from labguard.envs.runner import run_episode
from labguard.envs.scene import default_scene
from labguard.ir.schema import LabGuardIR
from labguard.runtime.rules import gold_irs, load_rules
from labguard.runtime.shield import SafetyShield, ShieldConfig


@pytest.fixture()
def scene():
    s = default_scene(np.random.default_rng(0))
    s.ppe = {"gloves", "goggles", "lab_coat"}
    return s


@pytest.fixture(scope="module")
def shield():
    return SafetyShield.from_rules(load_rules(), config=ShieldConfig(kinematic=False, risk=False))


def test_all_gold_irs_compile(examples, registry):
    monitors, errors = Compiler(registry).compile_many(e.ir for e in examples)
    assert not errors and len(monitors) == len(examples)


def test_default_rules_compile(registry):
    monitors, errors = Compiler(registry).compile_many(gold_irs(load_rules()))
    assert not errors and len(monitors) >= 24


def test_compile_rejects_bad_ir(registry):
    m, err = Compiler(registry).try_compile(
        LabGuardIR("temperature_constraint", ["C2H5OH"], "action.target_temperature <= ", ["heat"], "medium", "modify"))
    assert m is None and err is not None


def test_scope_and_binding(registry, scene):
    m, _ = Compiler(registry).try_compile(LabGuardIR(
        "temperature_constraint", ["C2H5OH"], "action.target_temperature <= 60", ["heat"], "medium", "modify"))
    hot = Action("heat", "flask_ethanol", "hot_plate_1", target_temperature=90.0)
    assert m.violated(scene, hot) is True
    assert m.violated(scene, Action("heat", "beaker_water", "hot_plate_1", target_temperature=90.0)) is None  # S
    assert m.violated(scene, Action("pour", "flask_ethanol", "beaker_target", volume=5.0)) is None  # T
    res = m(scene, hot)
    assert res.verdict == "modify" and res.action.target_temperature == pytest.approx(60.0)


def test_shield_block_modify_pass(shield, scene):
    d = shield.filter(scene, Action("pour", "bottle_hcl", "beaker_bleach", volume=20.0, pour_rate=10.0))
    assert d.verdict == "block" and d.action is None and "R01" in d.fired_rules
    d = shield.filter(scene, Action("pour", "beaker_water", "beaker_target", volume=40.0, pour_rate=60.0))
    assert d.verdict == "modify" and d.action.pour_rate <= 25.0
    d = shield.filter(scene, Action("pour", "beaker_water", "beaker_target", volume=40.0, pour_rate=15.0))
    assert d.verdict == "pass" and not d.fired_rules


def test_action_filter_clamps_workspace(scene):
    sh = SafetyShield.from_rules(load_rules(), config=ShieldConfig(semantic=False, risk=False))
    d = sh.filter(scene, Action("move", target_pos=np.array([1.5, 0.0, 0.3])))
    assert d.verdict in ("modify", "block")
    if d.action is not None:
        assert d.action.target_pos[0] < 1.0


def test_closed_loop_full_not_worse_than_base():
    full = SafetyShield.from_rules(load_rules(), config=ShieldConfig(planning=True))
    base = [run_episode("PourLiquid", None, seed=k) for k in range(6)]
    safe = [run_episode("PourLiquid", full, seed=k) for k in range(6)]
    assert sum(r.success for r in safe) >= sum(r.success for r in base)
    assert sum(bool(r.fatal_hazards) for r in safe) <= sum(bool(r.fatal_hazards) for r in base)


def test_risk_score_eq12(scene):
    from labguard.runtime.risk import RISK_WEIGHTS, RiskScorer

    rs = RiskScorer()
    assert (rs.w_v, rs.w_p, rs.w_c, rs.w_u) == RISK_WEIGHTS == (0.25, 0.25, 0.30, 0.20)
    mix = rs(scene, Action("pour", "bottle_hcl", "beaker_bleach", volume=20.0, meta={"uncertainty": 1.0}))
    assert mix.chemical == 1.0 and mix.total >= 0.5
    water = rs(scene, Action("pour", "beaker_water", "beaker_target", volume=5.0))
    assert water.chemical < 0.5 and water.total < mix.total
    hot = rs(scene, Action("heat", "flask_ethanol", "hot_plate_1", target_temperature=90.0))
    assert hot.chemical == 1.0  # above ethanol's safe heating limit
