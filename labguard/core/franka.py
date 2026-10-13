"""Franka Emika Panda kinematic model (modified/Craig DH parameters).

Used by the kinematic ActionFilter, the typed monitor repairs, the toy
simulator and the scripted policies.
"""

from __future__ import annotations

import numpy as np

# Modified DH (a_{i-1}, d_i, alpha_{i-1}) for joints 1..7 plus the flange.
PANDA_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
FLANGE_D = 0.107
GRIPPER_D = 0.1034  # flange -> fingertip centre (Franka Hand)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
QDOT_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _mdh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


class PandaKinematics:
    def __init__(self, tool_offset: float = GRIPPER_D) -> None:
        self.tool_offset = tool_offset

    def frames(self, q: np.ndarray) -> list[np.ndarray]:
        """Homogeneous transforms of joint frames 1..7 and the tool frame."""
        T = np.eye(4)
        out = []
        for (a, d, alpha), qi in zip(PANDA_DH, q):
            T = T @ _mdh(a, d, alpha, float(qi))
            out.append(T.copy())
        T = T @ _mdh(0.0, FLANGE_D + self.tool_offset, 0.0, 0.0)
        out.append(T)
        return out

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.frames(np.asarray(q, dtype=float))[-1]

    def position(self, q: np.ndarray) -> np.ndarray:
        return self.fk(q)[:3, 3]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6x7 geometric Jacobian of the tool frame (linear; angular)."""
        frames = self.frames(np.asarray(q, dtype=float))
        p_e = frames[-1][:3, 3]
        J = np.zeros((6, 7))
        for i in range(7):
            z = frames[i][:3, 2]
            p = frames[i][:3, 3]
            J[:3, i] = np.cross(z, p_e - p)
            J[3:, i] = z
        return J

    def ik_position(self, target: np.ndarray, q0: np.ndarray, iters: int = 200, tol: float = 1e-4,
                    damping: float = 0.05, max_step: float = 0.2) -> tuple[np.ndarray, bool]:
        """Damped-least-squares position IK (orientation free)."""
        q = np.asarray(q0, dtype=float).copy()
        target = np.asarray(target, dtype=float)
        for _ in range(iters):
            err = target - self.position(q)
            if np.linalg.norm(err) < tol:
                return q, True
            Jp = self.jacobian(q)[:3]
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + damping ** 2 * np.eye(3), err)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            # null-space pull toward home keeps the arm away from joint limits
            N = np.eye(7) - np.linalg.pinv(Jp) @ Jp
            dq += 0.05 * N @ (Q_HOME - q)
            q = np.clip(q + dq, Q_MIN, Q_MAX)
        return q, bool(np.linalg.norm(target - self.position(q)) < 10 * tol)

    def ee_speed(self, q: np.ndarray, qdot: np.ndarray) -> float:
        return float(np.linalg.norm(self.jacobian(q)[:3] @ np.asarray(qdot, dtype=float)))
