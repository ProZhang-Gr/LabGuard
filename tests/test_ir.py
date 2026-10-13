"""LabGuard-IR: schema, six-line serialization and the predicate language (Sec. 4.1)."""

from labguard.ir.predicate import canonicalize, predicate_tokens
from labguard.ir.schema import LabGuardIR, validate_ir
from labguard.ir.serialize import ir_to_text, text_to_ir


def test_gold_irs_are_schema_valid(examples, registry):
    bad = [(e.id, validate_ir(e.ir, registry)) for e in examples if validate_ir(e.ir, registry)]
    assert not bad, bad[:5]


def test_serialization_round_trip(examples, registry):
    for e in examples[::7]:
        text = ir_to_text(e.ir)
        assert len(text.splitlines()) == 6
        back = text_to_ir(text)
        assert back.normalized(registry).to_dict() == e.ir.normalized(registry).to_dict(), e.id


def test_canonical_predicate_equivalence(registry):
    assert canonicalize("80 < action.target_temperature", registry) == \
        canonicalize("action.target_temperature > 80", registry)
    assert canonicalize("not (action.speed > 0.25)", registry) == canonicalize("action.speed <= 0.25", registry)
    # Soft Match is surface-level (Sec. 5.3): the flipped form is not token-identical
    assert predicate_tokens("80 < action.target_temperature") != predicate_tokens("action.target_temperature > 80")


def test_invalid_irs_are_rejected(registry):
    ok = LabGuardIR("temperature_constraint", ["C2H5OH"], "action.target_temperature <= 60", ["heat"],
                    "medium", "modify")
    assert not validate_ir(ok, registry)
    for bad in (
        LabGuardIR("temperature_limit", ["C2H5OH"], "action.target_temperature <= 60", ["heat"], "medium", "modify"),
        LabGuardIR("temperature_constraint", ["unobtainium"], "action.target_temperature <= 60", ["heat"],
                   "medium", "modify"),
        LabGuardIR("temperature_constraint", ["C2H5OH"], "action.target_temperature <== 60", ["heat"],
                   "medium", "modify"),
        LabGuardIR("temperature_constraint", ["C2H5OH"], "boil(src)", ["heat"], "medium", "modify"),
        LabGuardIR("temperature_constraint", ["C2H5OH"], "action.target_temperature <= 60", ["fly"],
                   "medium", "modify"),
    ):
        assert validate_ir(bad, registry), bad
