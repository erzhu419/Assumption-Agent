"""End-to-end runtime checks with scripted providers, not discovery evidence.

The test author is deliberately given a simple affine procedure. These tests
exercise paid practice, publication and frozen/archive execution, not whether an
LLM discovers that procedure or whether it generalizes beyond affine apparatuses.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_anonymous_method_learning import protocol, run_learning


FITTED_MODEL = """def fit(rows):
    global slope, intercept
    xs = [r['features']['u'] for r in rows]
    ys = [r['target'] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / sum((x-mx)**2 for x in xs)
    intercept = my - slope * mx
def predict(row):
    return slope * row['features']['u'] + intercept
"""

LEARNED_METHOD = """import json
def select_actions(spec, records, memory):
    steps = json.load(open('research_state.json'))['method_results']
    prior = memory['research_roles']['predictors'][0]
    if not steps:
        return {'action': 'evaluate', 'arguments': {'model': prior}}
    last = steps[-1]
    if last['proposal'].get('action') == 'evaluate':
        feedback = last['feedback']
        if feedback.get('returncode') == 0 and feedback['mse'] <= 1e-10:
            return {'model': prior, 'lesson': 'The old predictor agrees with current paid evidence.'}
        return {'action': 'write', 'arguments': {'path': 'local_revision.py', 'code': __FITTED__}}
    if last['proposal'].get('action') == 'write':
        return {'model': 'local_revision.py', 'lesson': 'Current paid evidence required a local fit.'}
    return {'stop': 'The scripted fixture has no further repair.'}
""".replace("__FITTED__", repr(FITTED_MODEL))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read_json(path):
    return json.loads(path.read_text())


@pytest.fixture
def memoryless_source(tmp_path):
    """A complete fixture learned from two purchased responses, with fixed weights."""
    root, seed = tmp_path / "sources", 7139
    write_json(root / "protocol.json", {
        "cases": [{"id": "case_000", "kind": "memoryless", "seed": seed}],
        "arms": ["code_react", "hegel"],
    })
    lab = AnonymousHistoryLab(kind="memoryless", seed=seed)
    action = {"kind": "trajectory", "channels": [],
              "sequence": [{"u": -1.0}, {"u": 1.0}]}
    measured = lab.execute(action)
    low, high = [row["target"] for row in measured["rows"]]
    parameters = {"slope": (high - low) / 2., "intercept": (high + low) / 2.}
    source_model = """import json
parameters = json.load(open('parameters.json'))
def predict(row):
    return parameters['slope'] * row['features']['u'] + parameters['intercept']
"""
    files = {
        "model.py": source_model,
        "parameters.json": json.dumps(parameters),
        "observations.json": json.dumps(measured["rows"]),
        "model_observations.json": json.dumps(measured["rows"]),
        "laboratory_observations.json": json.dumps([measured]),
        "notes.md": "A fixture predictor fitted only to the two purchased endpoints.",
    }
    for arm in ("code_react", "hegel"):
        workspace = root / "case_000" / arm
        write_json(workspace / "archive.json", {
            "files": files, "lessons": ["Compare archived predictions with new paid evidence."],
            "question_traces": [],
        })
        write_json(workspace / "training.json", {
            "workspace": str(workspace), "model": "model.py", "status": "completed",
            "fresh_confirmation": {"passed": True, "mse": 0.0, "rows": 96},
            "events": [{"action": "query", "arguments": {"actions": [action]},
                        "feedback": {"observations": [measured]}}],
        })
    return root


def action(name, arguments):
    return {"action": name, "arguments": arguments,
            "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}


def test_practice_publication_frozen_transfer_and_equal_live_archive(memoryless_source, tmp_path):
    output = tmp_path / "study"
    plan = protocol(memoryless_source, model="scripted-interface-only")
    plan["author_limits"]["max_model_calls"] = 2
    plan["limits"]["max_model_calls"] = 1
    live_requests = []

    class ScriptedAuthor:
        def next_action(self, request):
            if not request["events"]:
                return action("write", {"path": METHOD_PATH, "code": LEARNED_METHOD})
            practice = request['events'][-1]['feedback']['edit_practice']
            assert practice['fresh_confirmation']['passed']
            assert practice['selected_prior_predictor']
            return action("finish", {"path": METHOD_PATH, "lesson": "Use paid errors to select reuse or refit."})

    class ProgramConsumer:
        def __init__(self, workspace):
            self.workspace = Path(workspace)

        def next_action(self, request):
            live_requests.append((self.workspace.name, deepcopy(request["session_knowledge"])))
            assert len(request["observations"]) == 72
            return action("run", {"operator": "procedure", "path": METHOD_PATH,
                                  "max_steps": 12, "confirm": True})

    def provider_factory(workspace):
        return ScriptedAuthor() if Path(workspace).name == "author" else ProgramConsumer(workspace)

    summary = run_learning(plan, output, provider_factory=provider_factory)

    assert summary["method_published"] and summary["author_status"] == "completed"
    author = read_json(output / "author" / "training.json")
    assert author["method_source"] == LEARNED_METHOD
    assert len(author["practice_trials"]) == 2
    assert [trial["selected_prior_predictor"] for trial in author["practice_trials"]] == [True, False]
    assert all(trial["fresh_confirmation"]["passed"] and trial["fresh_confirmation"]["rows"] == 96
               for trial in author["practice_trials"])
    assert summary["author_costs"]["observation_cost"] == 2 * (72 + 96)
    assert (output / "author" / "trials" / "trial_001" / "local_revision.py").is_file()

    initial = read_json(output / "initial_archive.json")
    transfer = read_json(output / "transfer_archive.json")
    assert transfer["files"]
    expected_files = {**initial["files"], METHOD_PATH: LEARNED_METHOD}
    assert transfer["files"] == expected_files
    assert "local_revision.py" not in transfer["files"]
    assert not any(name.startswith("trials/") for name in transfer["files"])
    assert {arm for arm, _ in live_requests} == {"code_react", "hegel"}
    assert len(live_requests) == 2
    assert all(memory == transfer for _, memory in live_requests)
    assert read_json(output / "controls" / "initial_archive.json") == transfer

    assert len(summary["frozen"]) == 3
    for result in summary["frozen"]:
        assert result["status"] == "completed" and result["terminal"]["passed"]
        assert result["terminal"]["mse"] < 1e-10
        assert result["model"] == "local_revision.py"
        assert result["costs"]["model_calls"] == result["costs"]["reported_tokens"] == 0
        assert result["costs"]["observation_cost"] == 72 + 96
        training = read_json(output / "frozen" / result["case"] / "training.json")
        assert training["fresh_confirmation"]["rows"] == 96
        assert training["method_source"] == LEARNED_METHOD
        assert training["method_results"][0]["proposal"]["action"] == "evaluate"
        assert training["method_results"][1]["proposal"]["action"] == "write"
    controls = summary["live_controls"]
    assert controls["evidence_kind"] == "scripted_interface_check"
    assert len(controls["trials"]) == 2
    for result in controls["trials"]:
        assert result["terminal"]["passed"] and result["fresh_confirmation_passed"]
        assert result["costs"]["model_calls"] == 1
        training = read_json(output / "controls" / result["case"] / result["arm"] / "training.json")
        assert training["fresh_confirmation"]["rows"] == 96
        assert training["events"][-1]["arguments"]["operator"] == "procedure"
    assert summary["unavailable_frozen_targets"] == summary["unavailable_live_episodes"] == 0


def test_unpublished_author_keeps_failures_and_leaves_all_targets_unavailable(memoryless_source, tmp_path):
    output = tmp_path / "failed_study"
    plan = protocol(memoryless_source, model="scripted-interface-only")
    plan["author_limits"]["max_model_calls"] = 1
    requested_workspaces = []

    class UnfinishedAuthor:
        def next_action(self, request):
            return action("finish", {"path": METHOD_PATH, "lesson": "This placeholder is unresolved."})

    def provider_factory(workspace):
        requested_workspaces.append(Path(workspace))
        assert Path(workspace).name == "author", "An unpublished method must not start live consumers."
        return UnfinishedAuthor()

    summary = run_learning(plan, output, provider_factory=provider_factory)

    assert not summary["method_published"]
    assert requested_workspaces == [output / "author"]
    assert summary["unavailable_frozen_targets"] == len(plan["transfer_cases"]) == 3
    assert summary["unavailable_live_episodes"] == 2 * len(plan["live_control_cases"]) == 2
    assert summary["frozen"] == [] and summary["live_controls"] is None
    assert summary["practice_trials"] == 1
    assert summary["author_costs"]["observation_cost"] == 72
    published = read_json(output / "published_method.json")
    assert not published["published"] and published["source"] is None
    assert published["practice_trials"][0]["status"] == "method_stopped"
    assert (output / "author" / "trials" / "trial_000" / "result.json").is_file()
    assert not (output / "transfer_archive.json").exists()
    assert not (output / "frozen").exists() and not (output / "controls").exists()
