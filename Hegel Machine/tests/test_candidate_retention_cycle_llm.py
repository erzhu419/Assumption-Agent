"""Frozen executable knowledge must survive a paid response change and return."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from hegel_machine.cognition.code_agent import archive_confirmed_python_model
from hegel_machine.cognition.online_inquiry import InquiryLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


def linear_episode(path, orientation, values):
    lab = InquiryLab(seed=8, mechanism="linear", driver=0, orientation=orientation)
    agent = TemporalOperatorAgent(None, path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=24, confirmation_rows=3,
            compute_budget=10000, execution_seconds=60.))
    first, second = lab.public_spec()["controls"]
    agent._execute("query", {"actions": [{"kind": "response", "point": {first: value, second: 0.}}
        for value in values]}, lab)
    return agent, lab


def dependencies(agent, lab, *, factor=1., bias=0.):
    for path, source in (("helper.py", f"factor = {factor!r}\n"),
                         ("params.json", json.dumps({"bias": bias}))):
        agent._execute("write", {"path": path, "code": source}, lab)


def fit_model(agent, lab, path):
    control = lab.public_spec()["controls"][0]
    source = ("import json\nfrom helper import factor\ncoefficient = None\n"
        "def fit(rows):\n global coefficient\n"
        f" coefficient = sum(row['target'] * row['features'][{control!r}] for row in rows) / "
        f"sum(row['features'][{control!r}] ** 2 for row in rows)\n"
        "def predict(row):\n"
        f" return factor * coefficient * row['features'][{control!r}] + json.load(open('params.json'))['bias']\n")
    agent._execute("write", {"path": path, "code": source}, lab)


def import_archive(archive, agent, candidate_id):
    prefix = Path("shared/candidates") / candidate_id
    shutil.copytree(archive["snapshot_workspace"], agent.workspace / prefix / "_evaluated_model")
    return str(prefix / archive["model"])


@pytest.mark.parametrize("evaluate_source", [False, True])
def test_confirmed_fit_and_dependencies_keep_original_context_through_change_and_return(
        tmp_path, evaluate_source):
    source, source_lab = linear_episode(tmp_path / "source", 1, (.1, .3, .5, .8))
    dependencies(source, source_lab)
    fit_model(source, source_lab, "models/source.py")
    source._prepare_program_state()
    source._save("research_context.json", json.dumps({"conversation": "source research history"}))
    selected = "models/source.py"
    if evaluate_source:
        feedback = source._execute("evaluate", {"model": selected}, source_lab)
        selected = feedback["evaluated_model"]["model"]
    assert source._execute("finish", {"model": selected}, source_lab)["fresh_confirmation"]["passed"]
    original = archive_confirmed_python_model(source, tmp_path / "bank/source/_evaluated_model")
    assert original["training_rows"] == 4
    assert not (Path(original["snapshot_workspace"]) / "research_context.json").exists()
    assert not (Path(original["snapshot_workspace"]) / "research_state.json").exists()
    assert (Path(original["snapshot_workspace"]) / "helper.py").read_text() == "factor = 1.0\n"
    assert json.loads((Path(original["snapshot_workspace"]) / "params.json").read_text()) == {"bias": 0.}
    source_rows = json.loads((Path(original["snapshot_workspace"]) / "model_observations.json").read_text())
    source_costs = deepcopy(source.costs)

    values = (.2, .4, .6, .9)
    changed, changed_lab = linear_episode(tmp_path / "changed", -1, values)
    # Same relative dependency names in the new task deliberately mean
    # something else. A frozen hypothesis must execute in its own old context.
    dependencies(changed, changed_lab, factor=9., bias=100.)
    old_path = import_archive(original, changed, "candidate_000")
    tested = changed._execute("evaluate", {"model": old_path}, changed_lab)
    expected_old_error = 4 * sum(value * value for value in values) / len(values)
    assert tested["mse"] == pytest.approx(expected_old_error)
    assert tested["evaluated_model"]["source_model"] == old_path
    assert source.costs == source_costs
    assert json.loads((changed.workspace / "_evaluated_model/model_observations.json").read_text()) == source_rows

    dependencies(changed, changed_lab)
    fit_model(changed, changed_lab, "models/revised.py")
    revised = changed._execute("evaluate", {"model": "models/revised.py"}, changed_lab)
    assert revised["mse"] < 1e-20
    assert changed._execute("finish", {"model": revised["evaluated_model"]["model"]},
        changed_lab)["fresh_confirmation"]["passed"]
    updated = archive_confirmed_python_model(changed, tmp_path / "bank/changed/_evaluated_model")
    assert updated["training_rows"] == 4

    returning, returning_lab = linear_episode(tmp_path / "return", 1, (.15, .35, .55, .85))
    dependencies(returning, returning_lab, factor=17., bias=200.)
    paths = [import_archive(archive, returning, f"candidate_{index:03d}")
             for index, archive in enumerate((original, updated))]
    first = returning._execute("evaluate", {"model": paths[0]}, returning_lab)
    assert first["mse"] < 1e-20
    second = returning._execute("evaluate", {"model": paths[1]}, returning_lab)
    assert second["mse"] > .1
    assert second["evaluated_model"]["source_model"] == paths[0]
    assert second["evaluated_model"]["mse"] < 1e-20
    final = returning._execute("finish", {"model": second["evaluated_model"]["model"]}, returning_lab)
    assert final["fresh_confirmation"]["passed"]
    assert source.costs == source_costs
    assert json.loads((returning.workspace / "_evaluated_model/model_observations.json").read_text()) == source_rows


def test_live_cycle_forks_one_source_then_continues_each_own_branch_before_terminal(tmp_path, monkeypatch):
    from scripts import run_candidate_retention_cycle_llm as runner

    plan = runner.protocol()
    plan["cases"] = plan["cases"][:1]
    first_requests, calls, finished, running = {}, {}, [], set()

    class FollowResearchChoice:
        def __init__(self, path):
            self.path = Path(path).resolve()
            calls[self.path] = 0

        def next_action(self, request):
            calls[self.path] += 1
            first_requests.setdefault(self.path, deepcopy(request))
            events = request["events"]
            if not events:
                models = [item["model"] for item in request["session_knowledge"]["candidates"]]
                arguments = ({"operator": "research_choice", "models": models} if models else
                             {"operator": "history", "base_lags": 2, "path": "models/learned.py"})
                decision = {"action": "run", "arguments": arguments}
            else:
                latest = events[-1]["feedback"]
                assert "error" not in latest, latest
                if latest.get("operator") == "history":
                    decision = {"action": "run", "arguments": {"operator": "research_choice",
                        "models": [latest["path"]]}}
                elif latest.get("operator") == "refit":
                    decision = {"action": "run", "arguments": {"operator": "research_choice"}}
                elif latest.get("next_action"):
                    decision = deepcopy(latest["next_action"])
                else:
                    assert latest.get("operator") == "research_choice"
                    decision = {"action": "run", "arguments": {"operator": "history",
                        "base_lags": 2, "path": "models/learned.py"}}
            return {**decision, "usage": {"total_tokens": 7, "input_tokens": 6,
                                           "cached_input_tokens": 3}}

    original_run = TemporalOperatorAgent.run
    original_execute = TemporalOperatorAgent._execute
    original_terminal = TemporalInquiryLab.evaluation_rows

    def run(self, laboratory, **kwargs):
        running.add(self.workspace)
        try:
            return original_run(self, laboratory, **kwargs)
        finally:
            running.remove(self.workspace)
            finished.append(self.workspace)

    def execute(self, action, arguments, laboratory):
        if action == "finish" or (action == "run" and arguments.get("operator")):
            assert self.workspace in running, "the host must not select research or finish"
        return original_execute(self, action, arguments, laboratory)

    terminal_calls = []

    def terminal(self, **kwargs):
        assert not running and len(finished) == 5
        terminal_calls.append(self)
        return original_terminal(self, **kwargs)

    monkeypatch.setattr(TemporalOperatorAgent, "run", run)
    monkeypatch.setattr(TemporalOperatorAgent, "_execute", execute)
    monkeypatch.setattr(TemporalInquiryLab, "evaluation_rows", terminal)
    output = tmp_path / "live"
    result = runner.run_study(output, provider_factory=FollowResearchChoice,
        study_protocol=plan, max_workers=1)
    assert result["attempted_episodes"] == 5 and len(calls) == len(terminal_calls) == 5
    assert result["registered_source_episodes"] == 1 and not result["skipped"]
    source = result["sources"][0]["episode"]
    assert source["confirmed_and_correct"]
    assert source["native_fit_operator_calls"] == {"history": 1}
    branches = {item["condition"]: item for item in result["branches"]}
    for condition, branch in branches.items():
        changed, returned = branch["stages"]
        assert changed["confirmed_and_correct"] and returned["confirmed_and_correct"]
        assert changed["native_fit_operator_calls"] == {}
        assert any(item["candidate_id"] == "candidate_000" and item["rejected_on_paid_rows"]
                   for item in changed["inherited_candidate_scores"])
        assert [item["candidate_id"] for item in changed["candidate_catalog"]] == ["candidate_000"]
        expected = ["candidate_000", "candidate_001"] if condition == "retained" else ["candidate_001"]
        assert [item["candidate_id"] for item in returned["candidate_catalog"]] == expected
        assert returned["selected_original_candidate"] == (condition == "retained")
        assert returned["native_fit_operator_calls"] == {}
        # Each branch's revised hypothesis must actually come from its own
        # changed episode, rather than two targets reset to a common source.
        revised = changed["published_candidate"]
        assert f"/{condition}/bank/" in revised["snapshot_workspace"]
        assert revised["created_position"] == 1
        assert revised["model"].startswith("shared/confirmed_models/candidate_001/")
        assert revised["declared_dependencies"] == []
        bank = Path(revised["snapshot_workspace"]) / "shared/confirmed_models"
        assert not (bank / "candidate_000").exists()
        assert not (Path(revised["snapshot_workspace"]) / "shared/candidate_catalog.json").exists()
        totals = result["conditions"][condition]["source_plus_target_costs"]
        for metric in ("observation_cost", "measurement_cost", "compute_cost", "model_calls", "reported_tokens"):
            assert totals[metric] == source["costs"][metric] + sum(stage["costs"][metric] for stage in branch["stages"])
        for stage in branch["stages"]:
            assert stage["costs"]["model_calls"] <= plan["limits"]["max_model_calls"]
            assert stage["costs"]["observation_cost"] <= plan["limits"]["observation_budget"]
            assert stage["costs"]["compute_cost"] <= plan["limits"]["compute_budget"]
            assert all(item["followed_immediately"] for item in stage["research_choice_recommendations"])
    assert result["actual_costs"]["model_calls"] == sum(calls.values())
    assert result["actual_costs"]["reported_tokens"] == 7 * sum(calls.values())
    roots = [path for path in first_requests if path.parent.name == "source"]
    assert len(roots) == 1
    case_dir = output / f"case_{plan['cases'][0]['apparatus_seed']}"
    for position in (1, 2):
        left = first_requests[(case_dir / "retained" / f"episode_{position:03d}").resolve()]
        right = first_requests[(case_dir / "latest_only" / f"episode_{position:03d}").resolve()]
        for key in ("limits", "observations", "laboratory_observations", "task_interface"):
            assert left[key] == right[key]


def test_failed_source_is_kept_and_dependent_targets_are_not_started(tmp_path):
    from scripts import run_candidate_retention_cycle_llm as runner

    plan = runner.protocol()
    plan["cases"] = plan["cases"][:1]
    created, calls = [], []

    class FailedSource:
        def __init__(self, path):
            self.path = Path(path)
            created.append(self.path)

        def next_action(self, request):
            calls.append(self.path)
            return {"action": "provider_error", "error": {"message": "source service failure"},
                    "usage": {"total_tokens": 11}}

    output = tmp_path / "failed_source"
    result = runner.run_study(output, provider_factory=FailedSource, study_protocol=plan, max_workers=1)
    assert len(created) == len(calls) == result["attempted_episodes"] == 1
    assert result["sources"][0]["episode"]["status"] == "provider_failed"
    assert result["branches"] == [] and len(result["skipped"]) == 4
    assert all(item["reason"] == "source_not_confirmed" for item in result["skipped"])
    assert result["actual_costs"]["model_calls"] == 1
    assert result["actual_costs"]["reported_tokens"] == 11
    assert result["service_available_matched_cases"] == []
    case_dir = output / f"case_{plan['cases'][0]['apparatus_seed']}"
    assert not (case_dir / "retained").exists() and not (case_dir / "latest_only").exists()
    with pytest.raises(FileExistsError):
        runner.run_study(output, provider_factory=FailedSource, study_protocol=plan)
    assert len(calls) == 1


def finish_payload(agent, lab, model, position, inherited, *, reused_files=()):
    class Finish:
        def next_action(self, request):
            return {"action": "finish", "arguments": {"model": model,
                "reused_files": list(reused_files)}, "usage": {"total_tokens": 1}}

    agent.provider = Finish()
    training = agent.run(lab)
    assert training["fresh_confirmation"]["passed"]
    return {"agent": agent, "training": training, "position": position,
            "path": agent.workspace, "inherited_candidates": inherited}


def test_published_composition_preserves_declared_candidate_dependency(tmp_path):
    from scripts import run_candidate_retention_cycle_llm as runner

    source, lab = linear_episode(tmp_path / "source", 1, (.1, .3, .5, .8))
    dependencies(source, lab)
    fit_model(source, lab, "models/source.py")
    original = runner.publish_candidate(finish_payload(source, lab, "models/source.py", 0, []),
                                        tmp_path / "source_bank")
    changed, changed_lab = linear_episode(tmp_path / "changed", -1, (.2, .4, .6, .9))
    shared = Path(runner.BANK_PATH) / original["candidate_id"] / "_evaluated_model"
    shutil.copytree(original["snapshot_workspace"], changed.workspace / shared)
    control = changed_lab.public_spec()["controls"][0]
    changed._execute("write", {"path": "models/composed.py", "code":
        "import json\n"
        f"rows = json.load(open({str(shared / 'model_observations.json')!r}))\n"
        f"slope = -sum(r['target'] * r['features'][{control!r}] for r in rows) / "
        f"sum(r['features'][{control!r}] ** 2 for r in rows)\n"
        f"def predict(row):\n return slope * row['features'][{control!r}]\n"}, changed_lab)
    payload = finish_payload(changed, changed_lab, "models/composed.py", 1, [original],
                             reused_files=[original["model"]])
    composed = runner.publish_candidate(payload, tmp_path / "changed_bank")
    assert composed["candidate_id"] == "candidate_001"
    assert composed["declared_dependencies"] == [runner.public_candidate(original)]
    bank = Path(composed["snapshot_workspace"])
    assert (bank / original["model"]).is_file()
    assert json.loads((bank / runner.CATALOG_PATH).read_text()) == [runner.public_candidate(original)]
    # A latest-only task that receives just the composed predictor must still
    # execute its explicitly declared dependency from the published context.
    consumer, consumer_lab = linear_episode(tmp_path / "consumer", -1, (.15, .35, .55, .85))
    destination = consumer.workspace / runner.BANK_PATH / composed["candidate_id"] / "_evaluated_model"
    shutil.copytree(composed["snapshot_workspace"], destination)
    feedback = consumer._execute("finish", {"model": composed["model"]}, consumer_lab)
    assert feedback["fresh_confirmation"]["passed"]


@pytest.mark.parametrize("revision", [None, "model", "binary_dependency"])
def test_same_inherited_path_gets_old_id_only_when_executable_context_is_unchanged(tmp_path, revision):
    import numpy as np
    from scripts import run_candidate_retention_cycle_llm as runner

    source, lab = linear_episode(tmp_path / "source", 1, (.1, .3, .5, .8))
    control = lab.public_spec()["controls"][0]
    code = ("import numpy as np\nslope = None\n"
        "def fit(rows):\n global slope\n"
        f" slope = sum(r['target'] * r['features'][{control!r}] for r in rows) / "
        f"sum(r['features'][{control!r}] ** 2 for r in rows)\n"
        f"def predict(row):\n return float(np.load('params.npy')) * slope * row['features'][{control!r}]\n")
    source._execute("write", {"path": "model.py", "code": code}, lab)
    np.save(source.workspace / "params.npy", 1.)
    original = runner.publish_candidate(finish_payload(source, lab, "model.py", 0, []),
                                        tmp_path / "source_bank")
    consumer, consumer_lab = linear_episode(tmp_path / "consumer", 1 if revision is None else -1,
                                            (.2, .4, .6, .9))
    root = consumer.workspace / runner.BANK_PATH / original["candidate_id"] / "_evaluated_model"
    shutil.copytree(original["snapshot_workspace"], root)
    if revision == "model":
        consumer._execute("write", {"path": original["model"],
            "code": code.replace("return float(np.load", "return -float(np.load")}, consumer_lab)
    elif revision == "binary_dependency":
        np.save(root / "params.npy", -1.)
    payload = finish_payload(consumer, consumer_lab, original["model"], 1, [original])
    published = runner.publish_candidate(payload, tmp_path / "consumer_bank")
    assert published["candidate_id"] == ("candidate_000" if revision is None else "candidate_001")
    assert payload["inherited_candidate_comparison"]["unchanged"] == (revision is None)
    assert bool(runner.selected_candidate(payload)) == (revision is None)
    assert float(np.load(Path(original["snapshot_workspace"]) / "params.npy")) == 1.
