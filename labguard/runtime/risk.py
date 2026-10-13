"""Diagnostic risk score (Eq. 12, Appendix B.5):

    rho_t = 0.25 rho_v + 0.25 rho_p + 0.30 rho_c + 0.20 rho_u

  rho_v  velocity: joint-speed utilisation |qdot| / qdot_max
  rho_p  proximity: closeness to joint limits, to the workspace boundary and of
         the end-effector target to containers not involved in the action
  rho_c  chemical hazard from the live container state: 1.0 when the action
         would combine an incompatible pair, otherwise the GHS hazard level of
         the involved substances, raised by the fill fraction after a transfer
         and by heating close to a substance's safe temperature limit
  rho_u  policy uncertainty reported by the policy (action.meta["uncertainty"])

The weights are fixed heuristics.  The score is logged and flags high-risk
steps (rho_t >= tau); it does not change verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from labguard.core.action import Action
from labguard.core.context import fill_after
from labguard.core.state import LabState
from labguard.ir.registry import Registry, default_registry
from labguard.runtime.kinematics import Q_MAX, Q_MIN, QDOT_MAX

RISK_WEIGHTS: tuple[float, float, float, float] = (0.25, 0.25, 0.30, 0.20)  # w_v, w_p, w_c, w_u


@dataclass
class RiskScore:
    total: float
    velocity: float
    proximity: float
    chemical: float
    uncertainty: float
    high: bool


@dataclass
class RiskScorer:
    w_v: float = RISK_WEIGHTS[0]
    w_p: float = RISK_WEIGHTS[1]
    w_c: float = RISK_WEIGHTS[2]
    w_u: float = RISK_WEIGHTS[3]
    tau: float = 0.6
    boundary_band: float = 0.05  # m, workspace-boundary / obstacle proximity band
    fill_band: tuple[float, float] = (0.8, 1.0)  # fill fraction mapped to [0, 1] overflow risk
    temp_band: float = 20.0  # degC below the safe limit at which heating risk starts
    dt: float = 0.05
    registry: Registry | None = None

    def __post_init__(self) -> None:
        self.registry = self.registry or default_registry()

    def _q_next(self, state: LabState, action: Action) -> np.ndarray:
        return action.q_target if action.q_target is not None else state.robot.q

    def velocity(self, state: LabState, action: Action) -> float:
        q = state.robot.q
        qdot = action.qdot if action.qdot is not None else (self._q_next(state, action) - q) / self.dt
        return float(np.clip(np.max(np.abs(qdot) / QDOT_MAX), 0.0, 1.0))

    def proximity(self, state: LabState, action: Action) -> float:
        half = (Q_MAX - Q_MIN) / 2.0
        q_next = self._q_next(state, action)
        margin = np.minimum(q_next - Q_MIN, Q_MAX - q_next) / half
        joint = float(np.clip(1.0 - margin.min(), 0.0, 1.0)) ** 2
        ws = obstacle = 0.0
        if action.target_pos is not None:
            p = np.asarray(action.target_pos, dtype=float)
            if "workspace" in state.zones:
                lo, hi = (np.asarray(b, dtype=float) for b in state.zones["workspace"])
                d = float(np.min(np.minimum(p - lo, hi - p)))
                ws = 1.0 if d < 0 else float(np.clip(1.0 - d / self.boundary_band, 0.0, 1.0))
            others = [c.position for cid, c in state.containers.items()
                      if cid not in (action.src, action.dst) and c.position is not None and cid != state.robot.holding]
            if others:
                d = min(float(np.linalg.norm((p - np.asarray(o, dtype=float))[:2])) for o in others)
                obstacle = float(np.clip(1.0 - d / self.boundary_band, 0.0, 1.0))
        return max(joint, ws, obstacle)

    def chemical(self, state: LabState, action: Action) -> float:
        reg = self.registry
        src = state.containers.get(action.src) if action.src else None
        dst = state.containers.get(action.dst) if action.dst else None
        s_src = src.substances if src else set()
        s_dst = dst.substances if dst else set()
        if any(not reg.compat(a, b) for a in s_src for b in s_dst):
            return 1.0
        hazard = max((reg.hazard_score(s) for s in s_src | s_dst), default=0.0)
        lo, hi = self.fill_band
        fill = fill_after(state, action)
        overflow = float(np.clip((fill - lo) / (hi - lo), 0.0, 1.0)) if fill is not None else 0.0
        heat = 0.0
        if src is not None:
            target = action.target_temperature if action.tag == "heat" else None
            temp = max(src.temperature, target if target is not None else src.temperature)
            for s in s_src:
                limit = reg.max_safe_temp(s)
                if limit is not None:
                    heat = max(heat, float(np.clip((temp - (limit - self.temp_band)) / self.temp_band, 0.0, 1.0)))
        return max(hazard, overflow, heat)

    def __call__(self, state: LabState, action: Action) -> RiskScore:
        rv = self.velocity(state, action)
        rp = self.proximity(state, action)
        rc = self.chemical(state, action)
        ru = float(np.clip(action.meta.get("uncertainty", 0.0), 0.0, 1.0))
        total = self.w_v * rv + self.w_p * rp + self.w_c * rc + self.w_u * ru
        return RiskScore(total, rv, rp, rc, ru, total >= self.tau)
