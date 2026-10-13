"""Scene construction for the toy laboratory (table-top Franka cell).

Layout in the robot base frame (metres), consistent with registry.DEFAULT_ZONES:

        y=+0.42  acid_cabinet | oxidizer_shelf | hot_zone(hot plate) | waste_area
                 general_shelf (x~0.26)     centrifuge      bench area
        y=-0.42  base_cabinet | flammable_cabinet |          | fume_hood
                 x=0.20 ...................................... x=0.72
"""

from __future__ import annotations

import numpy as np

from labguard.core.franka import Q_HOME, PandaKinematics
from labguard.core.state import Container, Equipment, LabState, RobotState

OBSTACLES = [
    ((0.00, 0.47, 0.00), (1.20, 0.90, 1.20)),  # left wall / shelving
    ((0.00, -0.90, 0.00), (1.20, -0.47, 1.20)),  # right wall
    ((0.78, -0.90, 0.00), (1.40, 0.90, 1.20)),  # back wall
]
CARRY_HEIGHT = 0.20


def default_scene(rng: np.random.Generator | None = None, jitter: float = 0.0) -> LabState:
    rng = rng or np.random.default_rng(0)

    def pos(x: float, y: float) -> np.ndarray:
        d = rng.uniform(-jitter, jitter, size=2) if jitter > 0 else np.zeros(2)
        return np.array([x + d[0], y + d[1], 0.0])

    containers = [
        Container("beaker_water", "beaker", {"H2O": 100.0}, 250.0, position=pos(0.45, -0.10), has_lid=False),
        Container("beaker_target", "beaker", {}, 250.0, position=pos(0.45, 0.08), has_lid=False),
        Container("bottle_hcl", "bottle", {"HCl": 200.0}, 500.0, lid_open=False, label="HCl",
                  position=pos(0.34, -0.20)),
        Container("beaker_bleach", "beaker", {"NaClO": 80.0}, 250.0, position=pos(0.56, 0.06), has_lid=False,
                  label="NaClO"),
        Container("flask_ethanol", "flask", {"C2H5OH": 100.0}, 250.0, position=pos(0.36, 0.12), label="EtOH"),
        Container("waste_aqueous", "waste_container", {}, 2000.0, label="aqueous_waste", lid_open=True,
                  position=np.array([0.63, 0.24, 0.0])),
        Container("waste_acid", "waste_container", {}, 2000.0, label="acid_waste", lid_open=True,
                  position=np.array([0.69, 0.24, 0.0])),
        Container("waste_halogenated", "waste_container", {}, 2000.0, label="halogenated_waste",
                  lid_open=True, position=np.array([0.63, 0.37, 0.0])),
        Container("waste_organic", "waste_container", {}, 2000.0, label="organic_waste", lid_open=True,
                  position=np.array([0.69, 0.37, 0.0])),
    ]
    equipment = [
        Equipment("hot_plate_1", "hot_plate", {"on": False, "calibrated": True, "clean": True},
                  position=(0.50, 0.28, 0.0), setpoint=22.0),
        Equipment("fume_hood_1", "fume_hood", {"on": True, "sash_lowered": True}, position=(0.62, -0.31, 0.0)),
        Equipment("centrifuge_1", "centrifuge", {"balanced": True, "lid_closed": True, "on": False},
                  position=(0.44, -0.22, 0.0)),
    ]
    kin = PandaKinematics()
    robot = RobotState(q=Q_HOME.copy(), ee_pos=kin.position(Q_HOME))
    return LabState(
        containers={c.id: c for c in containers},
        equipment={e.id: e for e in equipment},
        robot=robot,
        ppe={"gloves", "goggles", "lab_coat"},
        obstacles=list(OBSTACLES),
    )


def grasp_point(state: LabState, cid: str, height: float = 0.05) -> np.ndarray:
    return state.containers[cid].position + np.array([0.0, 0.0, height])


def above(state: LabState, eid: str, height: float = CARRY_HEIGHT) -> np.ndarray:
    ent = state.entity(eid)
    if ent is None:
        if state.is_zone(eid):
            return state.zone_center(eid) + np.array([0.0, 0.0, height])
        raise KeyError(eid)
    return ent.position + np.array([0.0, 0.0, height])


def put_in_gripper(state: LabState, cid: str, ee_target: np.ndarray, kin: PandaKinematics | None = None,
                   grip_force: float = 15.0) -> None:
    """Initialise an episode with the robot already holding `cid` at `ee_target`."""
    kin = kin or PandaKinematics()
    q, _ = kin.ik_position(ee_target, state.robot.q)
    state.robot.q = q
    state.robot.ee_pos = kin.position(q)
    state.robot.holding = cid
    state.robot.gripper_force = grip_force
    state.containers[cid].position = state.robot.ee_pos - np.array([0.0, 0.0, 0.05])
