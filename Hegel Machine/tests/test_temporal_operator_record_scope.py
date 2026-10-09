"""Current-regime fitting preserves, but does not mix in, prior paid evidence."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


def drifting_research(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, calibration_change={"after_rows": 72,
            "apparatus_seed": 99106007, "response_seed": 99106007})
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                               execution_seconds=30., compute_budget=200000))
    rng = Random(9607)
    actions = [{"kind": "trajectory", "sequence": [{"u": rng.uniform(-1., 1.)}
                for _ in range(24)], "channels": ["meter"]} for _ in range(6)]
    acquired = agent._execute("query", {"actions": actions}, lab)
    assert acquired["record_indices"] == list(range(6))
    assert [row for record in agent.observations for row in record["row_indices"]] == list(range(144))
    return agent, lab


def operator(agent, lab, name, **arguments):
    return agent._execute("run", {"operator": name, **arguments}, lab)


def test_selected_current_records_repair_drift_without_reusing_old_labels(tmp_path):
    agent, lab = drifting_research(tmp_path)
    history = deepcopy(agent.observations)
    paid = deepcopy(lab.costs)
    old = operator(agent, lab, "instrument", records=[0, 1], path="old.json")
    assert old["records"] == [0, 1] and old["support_rows"] == 48
    old_support = operator(agent, lab, "assess_instrument", instrument="old.json", records=[2])
    assert any(item["status"] == "supported" for item in old_support["hypotheses"])
    contradicted = operator(agent, lab, "assess_instrument", instrument="old.json", records=[3, 4, 5])
    assert all(item["status"] == "refuted" for item in contradicted["hypotheses"])

    current = operator(agent, lab, "instrument", records=[3, 4], path="current.json")
    assert current["records"] == [3, 4] and current["support_rows"] == 48
    assessed = operator(agent, lab, "assess_instrument", instrument="current.json", records=[5])
    survivors = [item for item in assessed["hypotheses"] if item["status"] == "supported"]
    assert len(survivors) == 1
    result = operator(agent, lab, "response", instrument="current.json", records=[3, 4, 5])
    assert result["split"] == {"records": [3, 4, 5], "training_records": [3],
                               "validation_records": [4, 5], "train_rows": 24,
                               "holdout_rows": 48}
    diagnostic = result["paid_meter_consistency"]
    assert diagnostic["records"] == [3, 4, 5]
    assert diagnostic["measured_rows"] == diagnostic["scored_rows"] == 72
    assert diagnostic["compute_cost"] == 72
    assert diagnostic["hypotheses"][0]["mse"] < 1e-10
    assert result["compute_cost"] == 12 * 24 + 48 + 12 * 72 + 72
    candidate = result["candidates"][0]
    assert candidate["train_mse"] < 1e-10
    assert candidate["validation_mse"] < 1e-10
    assert candidate["refit_train_mse"] < 1e-10
    source = (agent.workspace / candidate["path"]).read_text()
    namespace = {}
    exec(compile(source, candidate["path"], "exec"), namespace)
    fresh = lab.evaluation_rows(length=48, seed=980613)
    errors = [(namespace["predict"](row) - row.target) ** 2 for row in fresh]
    assert sum(errors) / len(errors) < 1e-10
    # Replaying the unchanged new instrument against excluded history exposes
    # the regime conflict that would contaminate its response diagnostics.
    old_diagnostic = agent._paid_meter_consistency(
        json.loads((agent.workspace / "current.json").read_text()), lab.public_spec(), [0, 1, 2])
    assert old_diagnostic["hypotheses"][0]["mse"] > 1e-4
    assert agent.observations == history and lab.costs == paid
    assert json.loads((agent.workspace / "laboratory_observations.json").read_text()) == json.loads(
        json.dumps(history))


def test_excluded_records_cannot_affect_fits_refits_or_validation_scope(tmp_path):
    agent, lab = drifting_research(tmp_path)
    first = operator(agent, lab, "instrument", records=[3, 4, 5], path="first.json")
    response = operator(agent, lab, "response", instrument="first.json", records=[3, 4, 5],
                        validation_records=[5], output_dir="shared/current")
    sources = {candidate["path"]: (agent.workspace / candidate["path"]).read_text()
               for candidate in response["candidates"]}
    # Deliberately change only excluded response and meter labels. Every fitted
    # candidate and its validation/consistency feedback must remain identical.
    for record in agent.observations[:3]:
        for row in record["rows"]:
            row["target"] += 100.
        for measurement in record["measurements"]:
            measurement["values"]["meter"] -= 100.
    second = operator(agent, lab, "instrument", records=[3, 4, 5], path="second.json")
    assert first["hypotheses"] == second["hypotheses"]
    assert first["compute_cost"] == second["compute_cost"]
    assert (agent.workspace / "first.json").read_text() == (agent.workspace / "second.json").read_text()
    repeated = operator(agent, lab, "response", instrument="second.json", records=[3, 4, 5],
                        validation_records=[5], output_dir="shared/current")
    assert response["split"] == repeated["split"]
    assert response["candidates"] == repeated["candidates"]
    assert response["paid_meter_consistency"] == repeated["paid_meter_consistency"]
    assert response["compute_cost"] == repeated["compute_cost"]
    assert sources == {path: (agent.workspace / path).read_text() for path in sources}
    before_compute = agent.costs["compute_cost"]
    with pytest.raises(ValueError, match="validation_records must be a subset"):
        operator(agent, lab, "response", instrument="second.json", records=[3, 4, 5],
                 validation_records=[2])
    assert agent.costs["compute_cost"] == before_compute
    with pytest.raises(ValueError, match="buy complete trajectories"):
        operator(agent, lab, "response", instrument="second.json", records=[])
    with pytest.raises(ValueError, match="choose paid records"):
        operator(agent, lab, "instrument", records=[])
    assert agent.costs["compute_cost"] == before_compute
