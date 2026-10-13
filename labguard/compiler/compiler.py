"""Schema-aware compiler: LabGuard-IR -> executable monitor (Sec. 3.4).

Compilation stages (an IR is rejected at the first failing stage):
  1. schema     closed-vocabulary fields (type, scope tags, severity, intervention)
  2. registry   every substance/entity mention resolves to a registry id
  3. predicate  predicate parses and only uses whitelisted names/functions with
                registry-known literals
  4. lowering   the predicate can be lowered into the typed monitor of type t
"""

from __future__ import annotations

from typing import Iterable

from labguard.compiler.library import MONITOR_LIBRARY
from labguard.compiler.monitor import CompiledMonitor
from labguard.ir.predicate import Predicate, PredicateError
from labguard.ir.registry import Registry, default_registry
from labguard.ir.schema import LabGuardIR, validate_ir

STAGES: tuple[str, ...] = ("schema", "registry", "predicate", "lowering")


class CompileError(ValueError):
    def __init__(self, stage: str, message: str, ir: LabGuardIR | None = None) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
        self.ir = ir


class Compiler:
    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or default_registry()

    def compile(self, ir: LabGuardIR) -> CompiledMonitor:
        errors = validate_ir(ir, self.registry)
        if errors:
            errors.sort(key=lambda e: STAGES.index(e[0]))
            stage = errors[0][0]
            raise CompileError(stage, "; ".join(m for s, m in errors if s == stage), ir)
        norm = ir.normalized(self.registry)
        try:
            pred = Predicate(norm.predicate, self.registry)
        except PredicateError as exc:  # pragma: no cover - validate_ir already parsed it
            raise CompileError("predicate", str(exc), ir) from exc
        cls = MONITOR_LIBRARY[norm.constraint_type]
        err = cls.lowering_error(pred)
        if err:
            raise CompileError("lowering", err, ir)
        return CompiledMonitor(norm, cls(norm, pred, self.registry))

    def try_compile(self, ir: LabGuardIR) -> tuple[CompiledMonitor | None, CompileError | None]:
        try:
            return self.compile(ir), None
        except CompileError as exc:
            return None, exc

    def compile_many(self, irs: Iterable[LabGuardIR]) -> tuple[list[CompiledMonitor], list[CompileError]]:
        monitors, errors = [], []
        for ir in irs:
            m, e = self.try_compile(ir)
            if m is not None:
                monitors.append(m)
            else:
                errors.append(e)  # type: ignore[arg-type]
        return monitors, errors


def compile_ir(ir: LabGuardIR, registry: Registry | None = None) -> CompiledMonitor:
    return Compiler(registry).compile(ir)
