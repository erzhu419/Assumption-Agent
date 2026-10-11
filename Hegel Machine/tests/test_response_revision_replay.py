import ast
from copy import deepcopy
import json

import pytest

from scripts import run_response_revision_replay as replay


def test_historical_declarations_preserve_current_execution_and_original_bundle():
    old = "ESTIMATORS = [{'id': 'earlier', 'rationale': 'source experience'}]\n"
    current = "import json\nESTIMATORS = [{'id': 'later'}]\ndef select_actions(memory):\n    return memory\n"
    bundle = {"files": {replay.matched.study.METHOD_PATH: current},
              "methods": [{"id": "retained_method"}]}
    before = deepcopy(bundle)
    revised = replay.historical_knowledge(bundle, old)
    assert bundle == before
    assert revised["methods"] == bundle["methods"]
    declarations, execution = replay.source_parts(revised["files"][replay.matched.study.METHOD_PATH])
    assert declarations == replay.source_parts(old)[0]
    assert ast.dump(ast.parse(execution)) == ast.dump(ast.parse(replay.source_parts(current)[1]))


def test_summary_reports_the_actual_revision_conditions_and_planned_size():
    summary = replay.matched.summarize([], 12, cases=("changed_mechanism", "aliased_shifted_mid"))
    assert summary["planned_runs"] == 12
    assert list(summary["conditions"]) == ["changed_mechanism", "aliased_shifted_mid"]
    assert summary["finished_runs"] == 0 and summary["status"] == "running"


def test_runtime_and_replay_runner_are_saved_before_live_workers_start(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "protocol.json").write_text(json.dumps({"limits": {}}))
    monkeypatch.setattr(replay.matched, "SOURCE", source)
    monkeypatch.setattr(replay.matched, "verify_codex_subscription", lambda: "test_no_provider")
    output = tmp_path / "cohort"

    class StopBeforeModel(Exception):
        pass

    class Pool:
        def __init__(self, **kwargs):
            for relative in ("scripts/run_response_revision_replay.py",
                    "src/hegel_machine/cognition/world_reference_hypothesis.py",
                    "src/hegel_machine/cognition/world_reference_recalibration.py"):
                assert (output / "frozen_runtime" / relative).read_text() == (
                    replay.ROOT / relative).read_text()
            raise StopBeforeModel

    monkeypatch.setattr(replay.matched, "ProcessPoolExecutor", Pool)
    bundle = {"files": {replay.matched.study.METHOD_PATH: "ESTIMATORS = []\n"}, "methods": []}
    with pytest.raises(StopBeforeModel):
        replay.matched.run(output, cases={"changed_mechanism": {}}, bundle=bundle)
