"""LabUtopia adapter on mock controller states (no Isaac Sim needed)."""

from types import SimpleNamespace

import numpy as np

from labguard.core.franka import Q_HOME
from labguard.envs.labutopia_adapter import LabUtopiaAdapterConfig, SafeLabUtopiaController, phase_to_tag


class MockController:
    """Mimics a LabUtopia controller: `current_phase` + step(state) -> (ArticulationAction, done, success)."""

    def __init__(self, phase: str, dq: np.ndarray) -> None:
        self.current_phase = phase
        self.dq = dq
        self.reset_called = False

    def step(self, state):
        q = np.asarray(state["joint_positions"], dtype=float)
        target = list(q[:7] + self.dq) + [0.04, 0.04]
        return SimpleNamespace(joint_positions=np.array(target, dtype=object), joint_velocities=None), False, False

    def reset(self):
        self.reset_called = True

    def need_reset(self):
        return False


def _state():
    return {"joint_positions": list(Q_HOME) + [0.04, 0.04], "object_name": "beaker_01",
            "object_position": [0.45, 0.0, 0.05], "target_position": [0.45, 0.2, 0.05], "target_name": "beaker_02"}


def test_phase_mapping():
    assert phase_to_tag("POURING") == "pour"
    assert phase_to_tag("move_to_target") == "transport"
    assert phase_to_tag(None, "PickController") == "pick"


def test_small_motion_passes():
    ctrl = SafeLabUtopiaController(MockController("pick", np.full(7, 1e-4)), config=LabUtopiaAdapterConfig())
    action, done, success = ctrl.step(_state())
    assert ctrl.steps == 1 and ctrl.interventions == 0
    assert not ctrl.need_reset()  # delegated to the wrapped controller


def test_fast_pour_is_slowed_down():
    dq = np.zeros(7)
    dq[6] = 0.5  # fast wrist rotation -> high pour rate
    ctrl = SafeLabUtopiaController(MockController("pour", dq), config=LabUtopiaAdapterConfig())
    action, _, _ = ctrl.step(_state())
    d = ctrl.decisions[-1]
    assert d.verdict == "modify" and "R06" in d.fired_rules  # pour-rate rule
    assert any(e.kind == "velocity_clamp" for e in d.filter_events)
    assert ctrl.interventions == 1
    q_cmd = np.array([float(x) for x in action.joint_positions[:7]])
    assert 0 < q_cmd[6] - Q_HOME[6] < 0.25  # the commanded 0.5 rad wrist step was reduced
    ctrl.reset()
    assert ctrl.controller.reset_called and ctrl.extractor.shadow is None
