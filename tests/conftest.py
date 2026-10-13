import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def registry():
    from labguard.ir.registry import default_registry

    return default_registry()


@pytest.fixture(scope="session")
def examples():
    from labguard.bench.io import load_examples

    return load_examples()


@pytest.fixture(scope="session")
def splits():
    from labguard.bench.io import load_all_splits

    return load_all_splits()
