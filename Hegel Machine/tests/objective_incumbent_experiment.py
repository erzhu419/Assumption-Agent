"""Load the rejected v049 experiment for its retained behavioral tests.

The live v048 default is intentionally unchanged. Only this private package
loads the frozen anticipatory implementation and its strict-comparison fallback.
"""

import importlib
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "artifacts/sc_kg_objective_incumbent_v049_20260911" /
          "anticipatory_comparison/method_source_anticipatory")
PACKAGE = "hegel_machine.cognition._incumbent_experiment_tests"


def _module(name):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SOURCE), str(ROOT / "src/hegel_machine/cognition")]
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.{name}")


ObjectiveResearch = _module("objective_research").ObjectiveResearch


def saved_experiment_optimizer(project_root, monkeypatch, case_id, **kwargs):
    import test_cognition_sc_kg_environment as sc_tests

    monkeypatch.setattr(sc_tests, "initialize_sc_kg_optimizer",
                        _module("sc_kg_environment").initialize_sc_kg_optimizer)
    return sc_tests._saved_terminal_failure_optimizer(project_root, monkeypatch, case_id, **kwargs)
