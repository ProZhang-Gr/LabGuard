"""Run a LabUtopia task with the LabGuard runtime in the control loop.

Copy (or symlink) this file into the LabUtopia repository root and run, e.g.

    python main_labguard.py --config-name level2_PourLiquid --headless \
        --labguard-config /path/to/labguard/configs/labutopia.yaml

All LabUtopia arguments are passed through unchanged.  The script wraps the
controller returned by LabUtopia's `create_controller` factory with
`SafeLabUtopiaController`; everything else (tasks, ACT inference, success
checks) is LabUtopia's own code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _split_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--labguard-config", default=None)
    parser.add_argument("--labguard-rules", default=None, help="runtime rule YAML (default: package rules)")
    parser.add_argument("--labguard-grounder", default=None,
                        help="grounder spec (e.g. 'regex', 'hybrid:<model>'); default: gold IR")
    parser.add_argument("--labguard-off", action="store_true", help="run the plain controller (Base)")
    parser.add_argument("--labguard-log", default=None, help="write the safety log (JSONL) here")
    ours, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]  # LabUtopia's parser rejects unknown arguments
    return ours


def main() -> None:
    ours = _split_args()
    sys.path.insert(0, os.getcwd())
    import main as lu_main  # LabUtopia's main.py: starts SimulationApp and imports its factories

    import yaml

    from labguard.envs.labutopia_adapter import LabUtopiaAdapterConfig, SafeLabUtopiaController
    from labguard.runtime.shield import SafetyShield, ShieldConfig

    cfg: dict = {}
    if ours.labguard_config:
        with open(ours.labguard_config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    grounder = None
    if ours.labguard_grounder:
        from labguard.grounder.factory import build_grounder

        grounder = build_grounder(ours.labguard_grounder)
    shield = SafetyShield.from_rules(ours.labguard_rules, grounder=grounder,
                                     config=ShieldConfig.from_dict(cfg.get("shield")))
    if shield.compile_errors:
        print(f"[LabGuard] {len(shield.compile_errors)} rule(s) rejected by the compiler:")
        for e in shield.compile_errors:
            print("   ", e)
    adapter_cfg = LabUtopiaAdapterConfig.from_dict(cfg.get("adapter"))
    wrapped: list[SafeLabUtopiaController] = []
    original_factory = lu_main.create_controller

    def create_controller(controller_type, cfg=None, robot=None, **kw):
        controller = original_factory(controller_type, cfg=cfg, robot=robot, **kw)
        if ours.labguard_off:
            return controller
        if cfg is not None and hasattr(cfg, "robot") and not cfg_has_base(adapter_cfg):
            adapter_cfg.robot_base = tuple(float(x) for x in cfg.robot.position)
        adapter_cfg.controller_type = str(controller_type)
        safe = SafeLabUtopiaController(controller, shield, adapter_cfg)
        wrapped.append(safe)
        return safe

    lu_main.create_controller = create_controller
    try:
        lu_main.main()
    finally:
        for safe in wrapped:
            print("[LabGuard]", json.dumps(safe.stats()))
            if ours.labguard_log:
                safe.shield.logger.dump(ours.labguard_log)


def cfg_has_base(adapter_cfg) -> bool:
    return any(abs(x) > 0 for x in adapter_cfg.robot_base)


if __name__ == "__main__":
    main()
