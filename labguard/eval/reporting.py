"""Markdown tables in the layout of the paper (Tables 2-5, 9) plus the paper's numbers for side-by-side comparison."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from labguard.eval.metrics import fmt_pm

# ----------------------------------------------------------------------------
# Paper reference numbers
# ----------------------------------------------------------------------------
PAPER_TABLE3: dict[str, tuple[int, int, int, float, float, float]] = {  # TP FN FP Prec Catch F1
    "No Safety": (7, 63, 1, 87.5, 10.0, 17.9),
    "Regex": (34, 36, 9, 79.1, 48.6, 60.2),
    "LoRA (Qwen3-8B)": (41, 29, 7, 85.4, 58.6, 69.5),
    "LabGuard-Hybrid": (41, 29, 7, 85.4, 58.6, 69.5),
    "Gold": (54, 16, 4, 93.1, 77.1, 84.4),
}

PAPER_TABLE5: dict[str, tuple[float, float, float, float, float, float, float]] = {  # SR IR VR Risk CR EL TCT
    "Base (no safety)": (10, math.nan, 2.8, 0.04, 1.9, 487, 24.1),
    "Planning only": (20, 0.8, 2.1, 0.05, 1.6, 512, 25.3),
    "Semantic only": (10, 1.2, 1.5, 0.05, 1.3, 495, 24.8),
    "Action Filter only": (30, 2.1, 0.9, 0.07, 0.7, 531, 26.5),
    "Risk Scoring only": (10, 0.3, 2.5, 0.12, 2.1, 476, 23.7),
    "Full LabGuard": (40, 3.2, 0.6, 0.08, 0.5, 548, 27.4),
}

PAPER_TABLE9: dict[str, tuple[float, float]] = {  # Base SR, Full SR (%)
    "pick": (94.3, 95.0), "place": (92.0, 93.3), "shake": (88.3, 90.0), "stir": (85.0, 87.7),
    "open": (91.7, 92.3), "close": (93.3, 94.0), "pour": (50.0, 56.7), "HeatLiquid": (33.3, 46.7),
    "PourLiquid": (45.0, 48.3), "ShakeBeaker": (72.3, 75.0), "TransportBeaker": (71.7, 78.3),
    "press": (96.7, 98.3), "HeatLiquidOOD": (30.0, 43.3), "CleanBeaker": (8.3, 15.0),
}


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(x: Any) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return str(x)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}"


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
GROUNDING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("type_acc", "Type Acc"), ("subst_f1", "Subst F1"), ("tag_f1", "Tag F1"), ("pred_soft", "Pred Soft"),
    ("pred_em", "Pred EM"), ("schema_valid", "Schema Valid"), ("compile_ready", "Compile Ready"),
)


def grounding_table(results: Mapping[str, Mapping[str, Any]],
                    columns: Sequence[tuple[str, str]] = GROUNDING_COLUMNS) -> str:
    """Table 2: `results` = {method: run_protocol(...) output}."""
    rows = [[name] + [fmt_pm(res["summary"][k]) if k in res["summary"] else "-" for k, _ in columns]
            for name, res in results.items()]
    return md_table(["Method"] + [c for _, c in columns], rows)


def stress_table(results: Mapping[str, Mapping[str, Any]], subset: str = "all", paper: bool = True) -> str:
    """Table 3: `results` = {config: run_stress(...) / run_stress_config(...) output}."""
    headers = ["Config", "TP", "FN", "FP", "Prec", "Catch", "F1"]
    rows = []
    for name, res in results.items():
        c = res["report"][subset]
        rows.append([name, c["TP"], c["FN"], c["FP"], pct(c["precision"]), pct(c["catch"]) + "%", pct(c["f1"])])
    out = md_table(headers, rows)
    if paper and subset == "all":
        out += "\n\nPaper (Table 3):\n\n" + md_table(
            headers, [[k, tp, fn, fp, f"{p:.1f}", f"{c:.1f}%", f"{f:.1f}"]
                      for k, (tp, fn, fp, p, c, f) in PAPER_TABLE3.items()])
    return out


def stress_category_table(res: Mapping[str, Any]) -> str:
    rows = [[cat, c["TP"], c["FN"], c["FP"], c["TN"], pct(c["catch"])]
            for cat, c in res["report"]["by_category"].items()]
    return md_table(["Category", "TP", "FN", "FP", "TN", "Catch"], rows)


def propagation_table(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Table 4: `results` = {method: stress_propagation(...) output}."""
    keys = (("tag_f1", "Tag F1"), ("pred_soft", "Pred Soft"), ("catch", "Catch"), ("precision", "Prec"),
            ("f1", "F1"))
    rows = [[name] + [fmt_pm(res["summary"][k]) for k, _ in keys] for name, res in results.items()]
    return md_table(["Method"] + [c for _, c in keys], rows)


def replay_table(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Replay diagnostics: `results` = {method: run_replay(...) output}."""
    keys = (("check_acc", "Check Acc"), ("scope_f1", "Scope F1"), ("param_recovery", "Param Rec"),
            ("decision_acc", "Decision Acc"), ("catch", "Catch"), ("precision", "Prec"), ("f1", "F1"),
            ("safety", "Safety"))
    rows = [[name] + [pct(res["summary"][k]) for k, _ in keys] for name, res in results.items()]
    return md_table(["Method"] + [c for _, c in keys], rows)


def ablation_table(res: Mapping[str, Any], paper: bool = True) -> str:
    """Table 5: `res` = run_ablation(...) output."""
    headers = ["Configuration", "SR", "IR", "VR", "Risk", "CR", "EL", "TCT"]
    rows = []
    for name, s in res["configs"].items():
        ir = s.get("IR_with_plan", s["IR"])
        rows.append([name, pct(s["SR"], 0) + "%", "–" if ir == 0 and name.startswith("Base") else f"{100 * ir:.1f}%",
                     f"{s['VR_offline']:.1f}", f"{s['Risk']:.2f}", f"{s['CR']:.1f}", f"{s['EL']:.0f}",
                     f"{s['TCT']:.1f}s"])
    out = md_table(headers, rows)
    if paper:
        out += "\n\nPaper (Table 5):\n\n" + md_table(headers, [
            [k, f"{sr:.0f}%", "–" if math.isnan(ir) else f"{ir:.1f}%", f"{vr:.1f}", f"{rk:.2f}", f"{cr:.1f}",
             f"{el:.0f}", f"{tct:.1f}s"] for k, (sr, ir, vr, rk, cr, el, tct) in PAPER_TABLE5.items()])
    return out


def runtime_table(res: Mapping[str, Any], base: str = "Base", full: str = "Full", paper: bool = True) -> str:
    """Table 9: `res` = run_runtime(...) output."""
    headers = ["Task", "Base SR", "Full SR", "ΔSR", "Full IR (%)", "Full VR", "Base unsafe"]
    if paper:
        headers += ["Paper Base SR", "Paper Full SR"]
    rows = []
    for task, r in res["tasks"].items():
        b, f = r[base], r[full]
        row = [f"{r['level']} {task}", f"{100 * b['SR']:.1f}±{100 * b['SR_std']:.1f}",
               f"{100 * f['SR']:.1f}±{100 * f['SR_std']:.1f}", f"{100 * (f['SR'] - b['SR']):+.1f}",
               f"{100 * f['IR']:.2f}%", f"{f['VR_logged']:.1f}", pct(b["unsafe_event_rate"])]
        if paper:
            pb, pf = PAPER_TABLE9.get(task, (math.nan, math.nan))
            row += [pb, pf]
        rows.append(row)
    return md_table(headers, rows)
