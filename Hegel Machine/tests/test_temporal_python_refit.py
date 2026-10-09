"""Original Python fitting procedures must be tested on separate paid trajectories."""

from copy import deepcopy
import json
from pathlib import Path
from random import Random
import shutil

import numpy as np
import pytest

from hegel_machine.cognition.code_agent import archive_confirmed_python_model
from hegel_machine.cognition.history_revision import revise_history
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_research_routing import revision_action
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


class LawLab(TemporalInquiryLab):
    """The real trajectory interface with an explicit test response law."""

    def __init__(self, *, slope=1., offset=0., quadratic=0.):
        super().__init__(task_index=1, apparatus_seed=71031, instrument_mode="finite2")
        self.slope, self.offset, self.quadratic = slope, offset, quadratic

    def execute(self, action):
        record = super().execute(action)
        for row in record["rows"]:
            u = row["features"]["u"]
            row["target"] = self.offset + self.slope * u + self.quadratic * u * u
        return record


MODEL = """import numpy as np
from fitting_dependency import paid_design
_coefficients = None
def fit(rows):
    global _coefficients
    design = paid_design(rows)
    _coefficients = np.linalg.lstsq(design, [row['target'] for row in rows], rcond=None)[0]
def predict(row):
    return float(np.asarray([1., row['features']['u']]) @ _coefficients)
"""

DEPENDENCY = """import json
from pathlib import Path
import numpy as np
def paid_design(rows):
    ordinary = json.load(open('observations.json'))
    fitting = json.load(open('model_observations.json'))
    batches = json.load(open('laboratory_observations.json'))
    assert ordinary == fitting == rows
    assert [row for batch in batches for row in batch['rows']] == rows
    for name in ('research_context.json', 'research_state.json', 'active.json', 'training.json'):
        assert not Path(name).exists(), name
    settings = json.load(open('settings.json'))
    assert settings['basis'] == 'linear'
    trace = Path('fit_trace.json')
    previous = json.loads(trace.read_text()) if trace.exists() else []
    previous.append({'rows': len(rows), 'targets': [row['target'] for row in fitting]})
    trace.write_text(json.dumps(previous))
    return np.asarray([[1., row['features']['u']] for row in ordinary])
"""


def make_agent(path, laboratory, *, compute_budget=10000, seed=81):
    agent = TemporalOperatorAgent(None, path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=192, confirmation_rows=12,
            execution_seconds=30., compute_budget=compute_budget))
    rng = Random(seed)
    agent._execute("query", {"actions": [
        {"kind": "trajectory", "channels": [],
         "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(12)]}
        for _ in range(3)]}, laboratory)
    return agent


def write_family(agent):
    for name, code in (("model.py", MODEL), ("fitting_dependency.py", DEPENDENCY),
                       ("settings.json", json.dumps({"basis": "linear"}))):
        agent._save(name, code)


def imported_family(tmp_path, target_lab):
    source_lab = LawLab()
    source = make_agent(tmp_path / "source", source_lab)
    write_family(source)
    submitted = source._execute("finish", {"model": "model.py"}, source_lab)
    assert submitted["fresh_confirmation"]["passed"]
    archive = archive_confirmed_python_model(source, tmp_path / "bank" / "_evaluated_model")
    target = make_agent(tmp_path / "target", target_lab, seed=93)
    destination = target.workspace / "shared" / "old" / "_evaluated_model"
    shutil.copytree(archive["snapshot_workspace"], destination)
    return target, "shared/old/_evaluated_model/model.py"


def content(path):
    return {str(item.relative_to(path)): item.read_bytes() for item in path.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts}


def refit(agent, laboratory, model="model.py", **arguments):
    return agent._execute("run", {"operator": "refit", "model": model,
        "path": "shared/refitted", "validation_records": [2], **arguments}, laboratory)


def test_old_coefficients_fail_but_original_fitting_family_adapts_without_mutating_archive(tmp_path):
    laboratory = LawLab(slope=-2., offset=.4)
    agent, model = imported_family(tmp_path, laboratory)
    old = agent._execute("evaluate", {"model": model}, laboratory)
    assert old["mse"] > .1
    archive = agent.workspace / Path(model).parent
    original, before = content(archive), deepcopy(agent.costs)
    result = refit(agent, laboratory, model)
    assert result["status"] == "refit_supported_on_validation"
    assert result["validation"]["fit_available"]
    assert result["validation"]["mse"] < 1e-20
    assert result["split"]["training_records"] == [0, 1]
    assert result["split"]["validation_records"] == [2]
    assert result["split"]["train_rows"] == 24 and result["split"]["holdout_rows"] == 12
    assert result["final_fitting_rows"] == result["paid_rows_at_refit"] == 36
    assert result["paid_records_at_refit"] == 3
    assert agent.costs["compute_cost"] - before["compute_cost"] == 36
    assert agent.costs["observation_cost"] == before["observation_cost"]
    assert content(archive) == original
    assert agent.model is None and not agent.confirmations

    candidate = agent.workspace / Path(result["path"]).parent
    validation_fit = json.loads((candidate / "fit_trace.json").read_text())[-1]
    expected = [row["target"] for record in agent.observations[:2] for row in record["rows"]]
    assert validation_fit == {"rows": 24, "targets": expected}
    assert len(json.loads((candidate / "model_observations.json").read_text())) == 36
    score = agent._execute("evaluate", {"model": result["path"]}, laboratory)
    assert score["mse"] < 1e-20
    assert json.loads((candidate / "fit_trace.json").read_text())[-1]["rows"] == 36
    assert agent.model is None
    submitted = agent._execute("finish", {"model": result["path"]}, laboratory)
    assert submitted["fresh_confirmation"]["passed"]


def test_underexpressive_fitting_family_remains_inadequate_on_whole_holdout(tmp_path):
    laboratory = LawLab(slope=.5, quadratic=3.)
    agent, model = imported_family(tmp_path, laboratory)
    result = refit(agent, laboratory, model)
    assert result["status"] == "refit_inadequate_on_validation"
    assert result["validation"]["fit_available"]
    assert result["validation"]["mse"] > .01
    assert result["split"]["validation_records"] == [2]
    assert result["path"] in agent.candidate_paths
    assert "general model-family rejection" in result["evidence_scope"]
    assert agent.model is None and not agent.confirmations


def test_current_model_dependencies_only_see_training_labels_during_validation(tmp_path):
    laboratory = LawLab(slope=3., offset=-.8)
    agent = make_agent(tmp_path, laboratory)
    write_family(agent)
    for filename in ("research_context.json", "research_state.json", "active.json", "training.json"):
        (agent.workspace / filename).write_text(json.dumps({"labels": [row.target for row in agent.rows]}))
    result = refit(agent, laboratory)
    assert result["status"] == "refit_supported_on_validation", result
    candidate = agent.workspace / Path(result["path"]).parent
    trace = json.loads((candidate / "fit_trace.json").read_text())
    assert len(trace) == 1 and trace[0]["rows"] == 24
    assert trace[0]["targets"] == [row.target for row in agent.rows[:24]]
    assert len(json.loads((candidate / "observations.json").read_text())) == 36
    assert not (candidate / "research_context.json").exists()
    assert (agent.workspace / "research_context.json").exists()


def test_missing_fit_reports_unavailable_without_executing_prediction(tmp_path):
    laboratory = LawLab()
    agent = make_agent(tmp_path, laboratory)
    agent._save("model.py", "from pathlib import Path\ndef predict(row):\n"
        " Path('prediction_ran').write_text('yes')\n return 0.\n")
    result = refit(agent, laboratory)
    assert result["status"] == "refit_unavailable" and result["path"] is None
    assert result["validation"]["fit_available"] is False
    assert "mse" not in result["validation"]
    assert not (agent.workspace / "shared/refitted/_evaluated_model/prediction_ran").exists()
    assert agent.model is None and not agent.candidate_paths


def test_insufficient_scoring_budget_does_not_copy_or_execute_refit(tmp_path):
    laboratory = LawLab()
    agent = make_agent(tmp_path, laboratory, compute_budget=35)
    write_family(agent)
    before = deepcopy(agent.costs)
    with pytest.raises(ValueError, match="compute budget exhausted"):
        refit(agent, laboratory)
    assert agent.costs["compute_cost"] == before["compute_cost"] == 0
    assert agent.costs["observation_cost"] == before["observation_cost"]
    assert not (agent.workspace / "shared/refitted").exists()
    assert not (agent.workspace / "fit_trace.json").exists()


def test_current_record_refit_repairs_real_response_shift_without_reviving_old_labels(tmp_path):
    source_lab = TemporalInquiryLab(task_index=1, apparatus_seed=71031,
        response_seed=71031, instrument_mode="finite2")
    source = make_agent(tmp_path / "source", source_lab)
    fitted = source._execute("run", {"operator": "history", "base_lags": 3, "path": "model.py"}, source_lab)
    assert source._execute("finish", {"model": fitted["path"]}, source_lab)["fresh_confirmation"]["passed"]
    archive = archive_confirmed_python_model(source, tmp_path / "bank" / "_evaluated_model")
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=71031, response_seed=71031,
        instrument_mode="finite2", calibration_change={"after_rows": 36, "response_seed": 71099})
    agent = make_agent(tmp_path / "target", lab, seed=93)
    shutil.copytree(archive["snapshot_workspace"], agent.workspace / "shared/old/_evaluated_model")
    model = "shared/old/_evaluated_model/model.py"
    rejected = agent._execute("finish", {"model": model}, lab)
    assert rejected["fresh_confirmation"]["passed"] is False
    rng = Random(131)

    def buy(count):
        agent._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
            "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(12)]} for _ in range(count)]}, lab)

    buy(2)
    assert len(agent.rows) == 72 and len(agent.observations) == 6
    all_history = refit(agent, lab, model, path="shared/global", validation_records=[5])
    assert all_history["status"] == "refit_inadequate_on_validation"
    assert all_history["validation"]["mse"] > 1e-4
    before = deepcopy(agent.costs)
    current = refit(agent, lab, model, path="shared/current", records=[3, 4, 5], validation_records=[5])
    assert current["status"] == "refit_supported_on_validation"
    assert current["validation"]["mse"] < 1e-20
    assert current["split"]["training_records"] == [3, 4]
    assert current["split"]["validation_records"] == [5]
    assert current["final_fitting_rows"] == 36 and current["paid_rows_at_refit"] == 72
    assert current["paid_records_at_refit"] == 6
    assert agent.costs["compute_cost"] - before["compute_cost"] == 36
    context = agent.workspace / Path(current["path"]).parent
    selected = agent.observations[3:6]
    expected_rows = [row for record in selected for row in record["rows"]]
    for filename in ("observations.json", "model_observations.json"):
        assert json.loads((context / filename).read_text()) == json.loads(json.dumps(expected_rows))
    assert json.loads((context / "laboratory_observations.json").read_text()) == json.loads(json.dumps(selected))
    # Old records still make the global diagnostic high, but are not newly acquired evidence.
    scored = agent._execute("evaluate", {"model": current["path"]}, lab)
    assert scored["mse"] > 1e-4
    advice = revision_action(agent, {"model": current["path"]}, lab.public_spec())
    assert advice["next_action"]["arguments"].get("operator") != "refit"

    buy(1)
    advice = revision_action(agent, {"model": current["path"]}, lab.public_spec())
    action = advice["next_action"]
    assert action["arguments"]["operator"] == "refit"
    assert action["arguments"]["records"] == [3, 4, 5, 6]
    again = agent._execute(action["action"], action["arguments"], lab)
    assert again["status"] == "refit_supported_on_validation"
    assert again["final_fitting_rows"] == 48 and again["paid_rows_at_refit"] == 84
    assert agent._execute("finish", {"model": again["path"]}, lab)["fresh_confirmation"]["passed"]
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"]
    assert agent.costs["model_calls"] == agent.costs["reported_tokens"] == 0


@pytest.mark.parametrize("records, message", [
    ([0, 0], "distinct"), ([0, 3], "distinct"), ([], "buy complete"), ([0, 1], "subset")])
def test_refit_rejects_invalid_record_selection_before_fitting(tmp_path, records, message):
    lab = LawLab()
    agent = make_agent(tmp_path, lab)
    write_family(agent)
    before = deepcopy(agent.costs)
    with pytest.raises(ValueError, match=message):
        refit(agent, lab, records=records)
    assert agent.costs["compute_cost"] == before["compute_cost"]
    assert agent.costs["observation_cost"] == before["observation_cost"]
    assert not (agent.workspace / "shared/refitted").exists()


def test_adequate_history_base_is_not_extended_for_floating_point_residue():
    rng, rows = np.random.default_rng(5), []
    for _ in range(6):
        controls = rng.uniform(-1., 1., 24)
        for index, control in enumerate(controls):
            history = tuple({"u": float(value)} for value in controls[:index])
            previous = controls[index - 1] if index else 0.
            older = controls[index - 2] if index > 1 else 0.
            target = .3 + .7 * control ** 2 - .4 * previous * older
            rows.append(Observation({"u": float(control)}, target, history))
    result = revise_history(rows, base_lags=2).resultdict
    assert result["revision"]["heldout_mse"] < result["base"]["heldout_mse"] < 1e-10
    assert result["status"] == "base_retained" and result["selected_lags"] == 2
