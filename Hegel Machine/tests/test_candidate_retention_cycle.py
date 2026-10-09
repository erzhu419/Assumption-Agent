"""The return must select measured old code rather than receive the old law."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_candidate_retention_cycle as study


def test_shared_source_change_and_measured_old_candidate_return(tmp_path, monkeypatch):
    def forbidden_terminal(*args, **kwargs):
        pytest.fail("terminal evidence requested before all training stopped")

    monkeypatch.setattr(study.TemporalInquiryLab, "evaluation_rows", forbidden_terminal)
    plan = study.protocol()
    development = {"apparatus_seed": 910061, "changed_response_seed": 910079, "seed": 910071}
    assert development["apparatus_seed"] not in {case["apparatus_seed"] for case in plan["cases"]}
    result = study.train_case(development, tmp_path / "case", plan)
    source = result["source"]["result"]
    retained, latest = ([episode["result"] for episode in result["arms"][arm]]
                        for arm in study.ARMS)
    assert source["fresh_confirmed"] and source["refit_count"] == 1
    assert retained[0]["initial_candidate_paths"] == latest[0]["initial_candidate_paths"] == ["shared/predictor_000.py"]
    assert retained[0]["paid_observations"] == latest[0]["paid_observations"]
    assert retained[0]["choices"][0]["evaluations"][0]["mse"] > 1e-4
    assert retained[0]["refit_count"] == latest[0]["refit_count"] == 1
    assert retained[0]["confirmed_model_files"] == latest[0]["confirmed_model_files"]
    assert retained[1]["initial_candidate_paths"] == ["shared/predictor_000.py", "shared/predictor_001.py"]
    assert latest[1]["initial_candidate_paths"] == ["shared/predictor_001.py"]
    assert retained[1]["paid_observations"] == latest[1]["paid_observations"]
    assert retained[1]["selected_original_source"] and retained[1]["refit_count"] == 0
    assert retained[1]["confirmed_model_files"] == source["confirmed_model_files"]
    assert retained[1]["choices"][0]["evaluations"][0]["mse"] < 1e-10
    assert latest[1]["refit_count"] == 1 and not latest[1]["selected_original_source"]
    for item in [source, *retained, *latest]:
        assert item["fresh_confirmed"]
        assert item["costs"]["observation_cost"] == 96
        assert item["costs"]["model_calls"] == 0
        assert item["training_wall_seconds"] >= item["costs"]["python_execution_seconds"]
        assert all(not path.startswith("_evaluated_model/") for path in item["confirmed_model_files"])
    assert retained[1]["costs"]["compute_cost"] < latest[1]["costs"]["compute_cost"]


def test_all_registered_training_precedes_terminal_and_failed_case_is_preserved(tmp_path, monkeypatch):
    plan = deepcopy(study.protocol())
    plan["cases"] = plan["cases"][:3]
    events = []

    def train(case, path, registered):
        assert json.loads((tmp_path / "run/protocol.json").read_text()) == registered
        events.append(("train", case["apparatus_seed"]))
        if case == registered["cases"][1]:
            raise RuntimeError("preserve this acquisition failure")
        return {"case": case, "directory": path}

    def score(item, registered):
        assert sum(event[0] == "train" for event in events) == 3
        events.append(("score", item["case"]["apparatus_seed"]))
        return item["case"]

    monkeypatch.setattr(study, "train_case", train)
    monkeypatch.setattr(study, "score_case", score)
    monkeypatch.setattr(study, "summarize", lambda results, failures, registered:
        {"registered_cases": len(registered["cases"]), "completed_cases": len(results), "failures": failures})
    result = study.run_study(tmp_path / "run", study_protocol=plan)
    assert result["registered_cases"] == 3 and result["completed_cases"] == 2
    assert len(result["failures"]) == 1
    assert "preserve this acquisition failure" in result["failures"][0]["error"]
    assert [event[0] for event in events] == ["train", "train", "train", "score", "score"]
    with pytest.raises(FileExistsError):
        study.run_study(tmp_path / "run", study_protocol=plan)
