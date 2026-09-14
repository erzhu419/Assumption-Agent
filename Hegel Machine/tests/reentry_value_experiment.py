"""Load the rejected v053 joint-action experiment for its behavior tests."""

import importlib
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/sc_kg_reentry_value_v053_20260911/method_source_v053"
PACKAGE = "hegel_machine.cognition._reentry_value_experiment_tests"


def _module(name):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SOURCE), str(ROOT / "src/hegel_machine/cognition")]
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.{name}")


ObjectiveResearch = _module("objective_research").ObjectiveResearch


def use_frozen_reentry_value_optimizer(monkeypatch):
    import test_cognition_sc_kg_environment as sc_tests

    monkeypatch.setattr(sc_tests, "initialize_sc_kg_optimizer",
                        _module("sc_kg_environment").initialize_sc_kg_optimizer)
