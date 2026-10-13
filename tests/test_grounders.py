"""Grounders (Sec. 4.2): rule-based, retrieval, grammar-constrained decoding, HF code path on a tiny LM."""

import pytest

from labguard.eval.metrics import grounding_scores
from labguard.grounder.constrained import IRGrammar
from labguard.grounder.factory import build_grounder
from labguard.ir.serialize import ir_to_text


def test_regex_grounder_outputs_schema_valid_irs(examples):
    g = build_grounder("regex")
    test = examples[::10]
    sc = grounding_scores(g.ground_batch([e.text for e in test]), [e.gold for e in test])
    assert sc["schema_valid"] >= 0.9
    assert sc["type_acc"] >= 0.5


def test_knn_grounder_on_random_split(examples, splits):
    train, test = splits["random"].select(examples)
    g = build_grounder("hybrid:knn")
    g.fit(train)
    sc = grounding_scores(g.ground_batch([e.text for e in test]), [e.gold for e in test])
    assert sc["type_acc"] >= 0.9 and sc["compile_ready"] >= 0.9


def test_grammar_accepts_gold_and_rejects_garbage(examples, registry):
    gr = IRGrammar(registry)
    for e in examples[::25]:
        text = ir_to_text(e.ir)
        assert gr.is_valid(text), text
        assert gr.check(text[: len(text) // 2]).ok
    bad = ir_to_text(examples[0].ir).replace("predicate=", "predicate====", 1)
    assert not gr.check(bad).ok
    assert not gr.check("type=not_a_type\n").ok


def test_hf_grounder_constrained_decoding_on_tiny_lm(examples, registry):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tiny_lm import tiny_model, tiny_tokenizer

    from labguard.grounder.hf import HFGrounder

    corpus = [ir_to_text(e.ir) for e in examples[:200]] + [e.text for e in examples[:200]]
    tok = tiny_tokenizer(corpus)
    model = tiny_model(len(tok))
    g = HFGrounder(model_obj=model, tokenizer_obj=tok, constrained=True, max_new_tokens=48, batch_size=2,
                   registry=registry)
    g.ground_batch([e.text for e in examples[:2]])
    gr = IRGrammar(registry)
    for raw in g.last_raw:
        assert gr.check(raw).ok, raw
