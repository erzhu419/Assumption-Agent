"""Load v051's unused extra nominations without enabling them in the default."""

import importlib
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/sc_kg_constraint_retention_v051_20260911/method_source_v051"
PACKAGE = "hegel_machine.cognition._constraint_nomination_experiment_tests"


def use_frozen_nomination_optimizer(monkeypatch):
    import test_cognition_sc_kg_environment as sc_tests

    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SOURCE), str(ROOT / "src/hegel_machine/cognition")]
        sys.modules[PACKAGE] = package
    module = importlib.import_module(f"{PACKAGE}.sc_kg_environment")
    monkeypatch.setattr(sc_tests, "initialize_sc_kg_optimizer", module.initialize_sc_kg_optimizer)
