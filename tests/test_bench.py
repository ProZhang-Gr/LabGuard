"""LabGuard-Bench statistics quoted in the paper (Sec. 3, Table 6)."""

from collections import Counter


def test_bench_size(examples):
    assert len(examples) == 812
    assert len({e.seed_id for e in examples}) == 203


def test_random_split(examples, splits):
    train, test = splits["random"].select(examples)
    assert (len(train), len(test)) == (649, 163)
    assert not {e.id for e in train} & {e.id for e in test}


def test_source_and_chemistry_splits(examples, splits):
    src = [n for n in splits if n.startswith("source_held_out:")]
    assert len(src) == 4
    for n in src:
        train, test = splits[n].select(examples)
        assert not {e.source for e in train} & {e.source for e in test}
    chem = [n for n in splits if n.startswith("chemistry_held_out:")]
    assert chem
    train, test = splits[chem[0]].select(examples)
    assert len({e.seed_id for e in test}) == 9
    assert not {e.seed_id for e in train} & {e.seed_id for e in test}


def test_fifteen_constraint_types(examples):
    assert len(Counter(e.ir.constraint_type for e in examples)) == 15
