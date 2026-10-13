"""LabGuard predicate language: the state/action predicate field p of LabGuard-IR.

A predicate is a *safety requirement* that must hold for a proposed action to
be safe; the monitor reports a violation when it evaluates to False.
Example (Table 1): ``not incompatible(src, dst)``.

Syntax is a restricted Python expression subset (parsed with ``ast`` and
interpreted by a whitelist evaluator -- never ``eval``):

  roots      src / dst (containers), action, robot, env
  attributes see ROOTS below
  functions  contains(c, 'HCl'), incompatible(c1, c2), hazard(c, 'flammable'),
             has_ppe('gloves'), done('open'), in_zone('fume_hood'),
             status('centrifuge', 'balanced'), distance('hot_plate'),
             abs(x), min(a, b, ...), max(a, b, ...)
  operators  and / or / not, < <= > >= == != in, + - * /

Evaluation is three-valued: if a referenced quantity is missing (e.g. the
action has no destination) the predicate is *not applicable* (None) and the
monitor does not fire.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from labguard.ir.registry import Registry

CONTAINER_ATTRS = frozenset(
    {"volume", "capacity", "temperature", "fill_ratio", "lid_open", "open_time", "label", "kind"})
ACTION_ATTRS = frozenset(
    {"volume", "speed", "pour_rate", "target_temperature", "grip_force", "tilt_angle", "duration",
     "fill_after", "tag"})
ROBOT_ATTRS = frozenset({"speed", "holding", "gripper_force"})
ENV_ATTRS = frozenset({"ambient_temperature", "time"})
ROOTS: dict[str, frozenset[str]] = {
    "src": CONTAINER_ATTRS,
    "dst": CONTAINER_ATTRS,
    "action": ACTION_ATTRS,
    "robot": ROBOT_ATTRS,
    "env": ENV_ATTRS,
}
CONTAINER_ROOTS = frozenset({"src", "dst"})

# name -> (min arity, max arity or None)
FUNCTIONS: dict[str, tuple[int, int | None]] = {
    "contains": (2, 2),
    "incompatible": (2, 2),
    "hazard": (2, 2),
    "has_ppe": (1, 1),
    "done": (1, 1),
    "in_zone": (1, 1),
    "status": (2, 2),
    "distance": (1, 1),
    "abs": (1, 1),
    "min": (2, None),
    "max": (2, None),
}
CONTAINER_ARGS: dict[str, tuple[int, ...]] = {"contains": (0,), "incompatible": (0, 1), "hazard": (0,)}
STRING_ARGS: dict[str, tuple[int, ...]] = {
    "contains": (1,), "hazard": (1,), "has_ppe": (0,), "done": (0,), "in_zone": (0,),
    "status": (0, 1), "distance": (0,),
}
SYMMETRIC_FUNCTIONS = frozenset({"incompatible"})
HAZARD_CLASSES = ("flammable", "oxidizer", "corrosive", "toxic", "water_reactive", "acid", "base",
                  "halogenated", "peroxide_former")

_OP_STR = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
           ast.In: "in", ast.NotIn: "not in"}
_FLIP = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Eq: ast.Eq,
         ast.NotEq: ast.NotEq}
_NEGATE = {ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt, ast.Eq: ast.NotEq,
           ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In}
NEGATE_OP_STR = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "==", "in": "not in",
                 "not in": "in"}


class PredicateError(ValueError):
    """Raised for predicates that are syntactically invalid or not compiler-acceptable."""


class _NotApplicable:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return "NA"


NA = _NotApplicable()


# ----------------------------------------------------------------------------
# Parsing & validation
# ----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    t = text.strip().strip("`").strip()
    t = t.rstrip(".;")
    for a, b in (("≤", "<="), ("≥", ">="), ("≠", "!="), ("&&", " and "), ("||", " or "),
                 ("“", "'"), ("”", "'"), ("‘", "'"), ("’", "'")):
        t = t.replace(a, b)
    t = re.sub(r"(?<![<>=!])=(?!=)", "==", t)
    t = re.sub(r"!(?!=)", " not ", t)
    t = re.sub(r"\bAND\b", "and", t)
    t = re.sub(r"\bOR\b", "or", t)
    t = re.sub(r"\bNOT\b", "not", t)
    t = re.sub(r"\btrue\b", "True", t)
    t = re.sub(r"\bfalse\b", "False", t)
    return re.sub(r"\s+", " ", t).strip()


def _check_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _check_node(node.body)
    elif isinstance(node, ast.BoolOp):
        for v in node.values:
            _check_node(v)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            raise PredicateError(f"unsupported unary operator {type(node.op).__name__}")
        _check_node(node.operand)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _OP_STR:
                raise PredicateError(f"unsupported comparison {type(op).__name__}")
        _check_node(node.left)
        for c in node.comparators:
            _check_node(c)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise PredicateError(f"unsupported arithmetic operator {type(node.op).__name__}")
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise PredicateError(f"unknown function {ast.unparse(node.func)!r}")
        if node.keywords:
            raise PredicateError("keyword arguments are not allowed")
        name = node.func.id
        lo, hi = FUNCTIONS[name]
        if len(node.args) < lo or (hi is not None and len(node.args) > hi):
            raise PredicateError(f"{name}() takes {lo}..{hi or 'n'} arguments, got {len(node.args)}")
        for k, arg in enumerate(node.args):
            if k in CONTAINER_ARGS.get(name, ()):
                if not (isinstance(arg, ast.Name) and arg.id in CONTAINER_ROOTS):
                    raise PredicateError(f"argument {k} of {name}() must be src or dst")
            elif k in STRING_ARGS.get(name, ()):
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                    raise PredicateError(f"argument {k} of {name}() must be a string literal")
            else:
                _check_node(arg)
    elif isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id not in ROOTS:
            raise PredicateError(f"unknown object in {ast.unparse(node)!r}")
        if node.attr not in ROOTS[node.value.id]:
            raise PredicateError(f"unknown attribute {node.value.id}.{node.attr}")
    elif isinstance(node, ast.Name):
        raise PredicateError(f"bare name {node.id!r} is not allowed (use an attribute or function)")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (bool, int, float, str)):
            raise PredicateError(f"unsupported constant {node.value!r}")
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            if not isinstance(e, ast.Constant):
                raise PredicateError("collections may only contain literals")
    else:
        raise PredicateError(f"unsupported syntax: {type(node).__name__}")


def parse_predicate(text: str) -> ast.Expression:
    if not isinstance(text, str) or not text.strip():
        raise PredicateError("empty predicate")
    norm = normalize_text(text)
    try:
        tree = ast.parse(norm, mode="eval")
    except SyntaxError as exc:
        raise PredicateError(f"syntax error: {exc.msg}") from exc
    _check_node(tree)
    return tree


def vocabulary_errors(tree: ast.AST, registry: "Registry") -> list[str]:
    """Check that literal arguments refer to known registry entries."""
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id  # type: ignore[attr-defined]
        lits = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if name == "contains" and registry.resolve(lits[0]) is None:
            errors.append(f"contains(): unknown substance {lits[0]!r}")
        elif name == "hazard" and lits[0] not in HAZARD_CLASSES:
            errors.append(f"hazard(): unknown hazard class {lits[0]!r}")
        elif name == "has_ppe" and lits[0] not in registry.ppe_items:
            errors.append(f"has_ppe(): unknown PPE item {lits[0]!r}")
        elif name == "in_zone" and lits[0] not in registry.zones:
            errors.append(f"in_zone(): unknown zone {lits[0]!r}")
        elif name == "done" and lits[0] not in registry.step_names():
            errors.append(f"done(): unknown step {lits[0]!r}")
        elif name == "distance" and registry.resolve(lits[0]) is None:
            errors.append(f"distance(): unknown entity {lits[0]!r}")
        elif name == "status":
            ent = registry.resolve(lits[0])
            if ent is None or ent not in registry.equipment_flags:
                errors.append(f"status(): unknown equipment {lits[0]!r}")
            elif lits[1] not in registry.equipment_flags[ent]:
                errors.append(f"status(): unknown flag {lits[1]!r} for {ent}")
    return errors


# ----------------------------------------------------------------------------
# Canonicalisation (used by Predicate Canonical EM and by the compiler)
# ----------------------------------------------------------------------------
def _const_value(node: ast.AST) -> Any:
    """Fold a constant arithmetic expression; raise ValueError if not constant."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const_value(node.operand)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _const_value(node.left), _const_value(node.right)
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)):
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div) and b != 0:
                return a / b
    raise ValueError("not constant")


def _is_const(node: ast.AST) -> bool:
    try:
        _const_value(node)
        return True
    except ValueError:
        return False


def _num(v: Any) -> Any:
    if isinstance(v, float) and not isinstance(v, bool):
        v = round(v, 6)
        if v.is_integer():
            return int(v)
    return v


class _Canonicalizer(ast.NodeTransformer):
    def __init__(self, registry: "Registry | None") -> None:
        self.registry = registry

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.Constant(_num(node.value))

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if _is_const(node):
            return ast.Constant(_num(_const_value(node)))
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.USub, ast.UAdd)) and _is_const(node):
            return ast.Constant(_num(_const_value(node)))
        if isinstance(node.op, ast.Not):
            inner = node.operand
            if isinstance(inner, ast.UnaryOp) and isinstance(inner.op, ast.Not):
                return inner.operand
            if isinstance(inner, ast.Compare) and len(inner.ops) == 1:
                return ast.Compare(inner.left, [_NEGATE[type(inner.ops[0])]()], inner.comparators)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) > 1:  # a < b < c  ->  a < b and b < c
            parts, left = [], node.left
            for op, right in zip(node.ops, node.comparators):
                parts.append(self.visit_Compare(ast.Compare(left, [op], [right])))
                left = right
            return self.visit_BoolOp(ast.BoolOp(ast.And(), parts))
        op, right = node.ops[0], node.comparators[0]
        left = node.left
        if _is_const(left) and not _is_const(right) and type(op) in _FLIP:
            left, right, op = right, left, _FLIP[type(op)]()
        if isinstance(right, ast.Constant) and isinstance(right.value, bool) \
                and isinstance(op, (ast.Eq, ast.NotEq)):
            positive = right.value == isinstance(op, ast.Eq)
            return left if positive else ast.UnaryOp(ast.Not(), left)
        return ast.Compare(left, [op], [right])

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        flat: list[ast.AST] = []
        for v in node.values:
            if isinstance(v, ast.BoolOp) and type(v.op) is type(node.op):
                flat.extend(v.values)
            else:
                flat.append(v)
        uniq: dict[str, ast.AST] = {}
        for v in flat:
            uniq.setdefault(ast.unparse(v), v)
        values = [uniq[k] for k in sorted(uniq)]
        if len(values) == 1:
            return values[0]
        return ast.BoolOp(node.op, values)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = node.func.id  # type: ignore[attr-defined]
        args = list(node.args)
        for k in STRING_ARGS.get(name, ()):
            lit = args[k].value  # type: ignore[attr-defined]
            if name in ("contains", "distance") or (name == "status" and k == 0):
                resolved = self.registry.resolve(lit) if self.registry else None
                lit = resolved or lit
            else:
                lit = lit.strip().lower().replace(" ", "_")
            args[k] = ast.Constant(lit)
        if name in SYMMETRIC_FUNCTIONS:  # canonical order: src before dst
            args = sorted(args, key=lambda a: {"src": "0", "dst": "1"}.get(ast.unparse(a), ast.unparse(a)))
        if name in ("min", "max"):
            args = sorted(args, key=ast.unparse)
        return ast.Call(node.func, args, [])


# ----------------------------------------------------------------------------
# Three-valued evaluator
# ----------------------------------------------------------------------------
def _truth(v: Any) -> Any:
    if v is NA or v is None:
        return NA
    return bool(v)


def _compare(op: ast.cmpop, a: Any, b: Any) -> bool:
    if isinstance(op, ast.Lt):
        return a < b
    if isinstance(op, ast.LtE):
        return a <= b
    if isinstance(op, ast.Gt):
        return a > b
    if isinstance(op, ast.GtE):
        return a >= b
    if isinstance(op, ast.Eq):
        return a == b
    if isinstance(op, ast.NotEq):
        return a != b
    if isinstance(op, ast.In):
        return a in b
    if isinstance(op, ast.NotIn):
        return a not in b
    raise PredicateError(f"bad operator {op}")


def _evaluate(node: ast.AST, ns: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, ns)
    if isinstance(node, ast.BoolOp):
        saw_na = False
        is_and = isinstance(node.op, ast.And)
        for v in node.values:
            r = _truth(_evaluate(v, ns))
            if r is NA:
                saw_na = True
            elif r is (not is_and):  # False short-circuits `and`, True short-circuits `or`
                return r
        return NA if saw_na else is_and
    if isinstance(node, ast.UnaryOp):
        v = _evaluate(node.operand, ns)
        if v is NA or v is None:
            return NA
        if isinstance(node.op, ast.Not):
            return not bool(v)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, ns)
        for op, comp in zip(node.ops, node.comparators):
            right = _evaluate(comp, ns)
            if left is NA or right is NA or left is None or right is None:
                return NA
            try:
                if not _compare(op, left, right):
                    return False
            except TypeError:
                return NA
            left = right
        return True
    if isinstance(node, ast.BinOp):
        a, b = _evaluate(node.left, ns), _evaluate(node.right, ns)
        if a is NA or b is NA or a is None or b is None:
            return NA
        try:
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            return a / b
        except (TypeError, ZeroDivisionError):
            return NA
    if isinstance(node, ast.Call):
        name = node.func.id  # type: ignore[attr-defined]
        fn = ns["__functions__"].get(name)
        if fn is None:
            return NA
        args = []
        for k, arg in enumerate(node.args):
            if k in CONTAINER_ARGS.get(name, ()):
                val = ns.get(arg.id)  # type: ignore[attr-defined]
            else:
                val = _evaluate(arg, ns)
            if val is NA or val is None:
                return NA
            args.append(val)
        out = fn(*args)
        return NA if out is None else out
    if isinstance(node, ast.Attribute):
        obj = ns.get(node.value.id)  # type: ignore[attr-defined]
        if obj is None:
            return NA
        val = getattr(obj, node.attr, None)
        return NA if val is None else val
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(e.value for e in node.elts)  # type: ignore[attr-defined]
    raise PredicateError(f"cannot evaluate {type(node).__name__}")


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Comparison:
    """A `path op constant` atom extracted from a predicate (monitor parameter theta)."""

    path: str
    op: str
    value: Any


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|<|>|\+|-|\*|/")


def predicate_tokens(text: str) -> list[str]:
    """Token sequence used by Predicate Soft Match (surface-level, not canonical)."""
    return _TOKEN_RE.findall(text or "")


class Predicate:
    """Parsed, validated predicate with canonical form and evaluator."""

    def __init__(self, text: str, registry: "Registry | None" = None) -> None:
        self.source = text
        self.registry = registry
        self.tree = parse_predicate(text)
        canon = _Canonicalizer(registry).visit(ast.parse(ast.unparse(self.tree), mode="eval"))
        self.canonical_tree: ast.Expression = ast.fix_missing_locations(canon)  # type: ignore[assignment]

    @classmethod
    def try_parse(cls, text: str, registry: "Registry | None" = None) -> "Predicate | None":
        try:
            return cls(text, registry)
        except PredicateError:
            return None

    def __repr__(self) -> str:
        return f"Predicate({self.canonical!r})"

    @cached_property
    def canonical(self) -> str:
        return ast.unparse(self.canonical_tree)

    def vocabulary_errors(self) -> list[str]:
        return vocabulary_errors(self.canonical_tree, self.registry) if self.registry else []

    def evaluate(self, namespace: Mapping[str, Any]) -> bool | None:
        """True = requirement holds, False = violation, None = not applicable."""
        out = _truth(_evaluate(self.canonical_tree, namespace))
        return None if out is NA else bool(out)

    # -- structural queries used by the compiler --------------------------------
    @cached_property
    def calls(self) -> list[tuple[str, tuple[Any, ...]]]:
        out = []
        for node in ast.walk(self.canonical_tree):
            if isinstance(node, ast.Call):
                args = tuple(a.id if isinstance(a, ast.Name) else _const_or_src(a) for a in node.args)
                out.append((node.func.id, args))  # type: ignore[attr-defined]
        return out

    def call_names(self) -> set[str]:
        return {c[0] for c in self.calls}

    def call_args(self, name: str) -> list[tuple[Any, ...]]:
        return [args for fn, args in self.calls if fn == name]

    @cached_property
    def attributes(self) -> set[str]:
        return {f"{n.value.id}.{n.attr}" for n in ast.walk(self.canonical_tree)
                if isinstance(n, ast.Attribute)}  # type: ignore[attr-defined]

    @cached_property
    def comparisons(self) -> list[Comparison]:
        out: list[Comparison] = []

        def walk(node: ast.AST, negated: bool) -> None:
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                walk(node.operand, not negated)
                return
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                left, right = node.left, node.comparators[0]
                if not _is_const(left) and _is_const(right):
                    op = _OP_STR[type(node.ops[0])]
                    out.append(Comparison(ast.unparse(left), NEGATE_OP_STR[op] if negated else op,
                                          _num(_const_value(right))))
                    return
            for child in ast.iter_child_nodes(node):
                walk(child, negated)

        walk(self.canonical_tree.body, False)
        return out

    def limits_on(self, *paths: str) -> list[Comparison]:
        return [c for c in self.comparisons if c.path in paths]

    def tokens(self) -> list[str]:
        return predicate_tokens(self.source)


def _const_or_src(node: ast.AST) -> Any:
    try:
        return _const_value(node)
    except ValueError:
        return ast.unparse(node)


def canonicalize(text: str, registry: "Registry | None" = None) -> str | None:
    p = Predicate.try_parse(text, registry)
    return p.canonical if p else None
