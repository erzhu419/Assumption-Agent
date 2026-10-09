"""The reused finite grammar proposes bases from public reference readings only."""

import ast
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition import world_reference_expansion
from hegel_machine.cognition.world_reference_expansion import execute_reference_expansion


PATH = "shared/consolidated_method.py"


def basis(feature):
    x = Program("feature", name=feature)
    return [x.to_dict(), Program("mul", children=(x, x)).to_dict()]


def same_basis(left, right):
    return len(left) == len(right) and all(term in right for term in left)


def public_table():
    return {"references": [{"crystal": i,
        "frequency": {"value": 100 + 3 * i + 2 * i * i, "evidence": i},
        "measurements": {"signal_a": {"value": float(i), "evidence": 10 + i},
                         "signal_b": {"value": float(10 - i), "evidence": 20 + i}}}
        for i in (1, 2, 3)],
        "targets": [{"crystal": 4, "measurements": {
            "signal_a": {"value": 4., "evidence": 14},
            "signal_b": {"value": 6., "evidence": 24}}}]}


def setup_agent(tmp_path, estimators=(), table=None):
    agent = SimpleNamespace(workspace=tmp_path, files={},
                            costs={"research_compute_units": 7, "python_execution_seconds": 3.})
    (tmp_path / "shared").mkdir()
    (tmp_path / PATH).write_text("ESTIMATORS = " + repr(tuple(estimators)) + "\n\ndef step():\n    return 'unchanged'\n")
    (tmp_path / "shared/reference_recalibration.json").write_text(json.dumps({
        "research_table": table if table is not None else public_table()}))
    return agent


def run(agent):
    return execute_reference_expansion(agent, object(), {"path": PATH})


def retained(agent):
    return ast.literal_eval(ast.parse((agent.workspace / PATH).read_text()).body[0].value)


def test_all_measured_features_receive_the_same_affine_and_quadratic_search(tmp_path):
    agent = setup_agent(tmp_path)
    result = run(agent)
    declarations = result["generated_hypotheses"]
    assert result["expanded_features"] == ["signal_a", "signal_b"]
    assert result["status"] == "untested"
    assert len(declarations) == 4
    for feature in ("signal_a", "signal_b"):
        assert any(item["basis"] == basis(feature)[:1] for item in declarations)
        assert any(same_basis(item["basis"], basis(feature)) for item in declarations)
    assert all(item["origin"] == "finite_grammar_search/public_calibration/old_refit_composition"
               for item in declarations)
    assert agent.costs["research_compute_units"] == 7 + result["compute_cost"]
    assert result["compute_cost"] > 0
    assert (agent.workspace / PATH).read_text().endswith("def step():\n    return 'unchanged'\n")
    state = json.loads((agent.workspace / "shared/reference_recalibration.json").read_text())
    assert state["basis_expansions"] == [result]
    assert state["research_table"] == public_table()


def test_existing_basis_is_preserved_and_not_redeclared_even_in_another_order(tmp_path):
    original = [{"id": "kept_affine", "basis": basis("signal_a")[:1]},
                {"id": "kept_quadratic", "basis": list(reversed(basis("signal_a")))}]
    agent = setup_agent(tmp_path, original)
    result = run(agent)
    assert list(retained(agent)[:2]) == original
    assert len(result["generated_hypotheses"]) == 2
    assert all(item["id"].startswith("grammar_signal_b_") for item in result["generated_hypotheses"])


def test_finite_search_time_is_added_to_the_existing_python_budget(tmp_path, monkeypatch):
    agent = setup_agent(tmp_path)
    clock = iter([10., 10.25, 11., 11.5])
    monkeypatch.setattr(world_reference_expansion, "monotonic", lambda: next(clock))
    run(agent)
    assert agent.costs["python_execution_seconds"] == 3.75
    run(agent)
    assert agent.costs["python_execution_seconds"] == 3.75


def test_retry_does_not_repeat_computation_or_source_changes(tmp_path):
    agent = setup_agent(tmp_path)
    first = run(agent)
    source = (agent.workspace / PATH).read_text()
    state = (agent.workspace / "shared/reference_recalibration.json").read_text()
    second = run(agent)
    assert second["generated_hypotheses"] == []
    assert second["expanded_features"] == []
    assert second["compute_cost"] == 0
    assert (agent.workspace / PATH).read_text() == source
    assert (agent.workspace / "shared/reference_recalibration.json").read_text() == state
    assert agent.costs["research_compute_units"] == 7 + first["compute_cost"]
    # A new world's reference table has no expansion markers, but its source
    # still contains the transferred declarations. It must not duplicate them.
    (agent.workspace / "shared/reference_recalibration.json").write_text(json.dumps({"research_table": public_table()}))
    assert run(agent)["generated_hypotheses"] == []
    assert (agent.workspace / PATH).read_text() == source


@pytest.mark.parametrize("missing", ["frequency", "reference_measurements", "target_measurements"])
def test_incomplete_public_calibration_does_not_invent_measurements(tmp_path, missing):
    table = public_table()
    if missing == "frequency":
        table["references"][0].pop("frequency")
    elif missing == "reference_measurements":
        table["references"][0]["measurements"] = {}
    else:
        table["targets"][0]["measurements"] = {}
    agent = setup_agent(tmp_path, table=table)
    result = run(agent)
    assert result["generated_hypotheses"] == []
    assert result["expanded_features"] == []
    assert result["compute_cost"] == 0


def test_model_measurement_definition_is_transferred_into_new_bases(tmp_path):
    measurement = {"instrument": "phase meter", "pattern": r"phase signal = ([\d.]+)", "group": 1}
    original = {"id": "original_signal_a", "basis": basis("signal_a")[:1],
                "measurements": {"signal_a": measurement}}
    agent = setup_agent(tmp_path, [deepcopy(original)])
    result = run(agent)
    learned = next(item for item in result["generated_hypotheses"]
                   if same_basis(item["basis"], basis("signal_a")))
    assert learned["measurements"] == {"signal_a": measurement}
    assert retained(agent)[0] == original
