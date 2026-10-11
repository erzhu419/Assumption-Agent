"""Source review and publication reuse existing paid-evidence/runtime contracts."""

import json
from pathlib import Path
import shutil

import pytest

from scripts import consolidate_response_measurement as study
from scripts.run_response_measurement_recovery import prepared_conflict, signal_hypothesis, submit
from test_discoveryworld_agent import LIMITS, ScriptedProvider


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("consolidation_source")
    agent, lab, _, _ = prepared_conflict(root / "agent")
    study.write_json(root / "prefix.json", {"observations": agent.observations})
    result = submit(agent, lab, signal_hypothesis())
    assert result["status"] == "completed"
    q = agent._question({"id": "reference_extension_5", "parent_id": "task",
                         "question": "Does the proposed setting activate the target?"})
    q.update(status="inconclusive", evidence=[len(agent.observations) - 1], answer="Not reviewed")
    agent._save_state()
    study.write_json(root / "result.json", {"agent": agent.result()})
    study.write_json(root / "protocol.json", {"limits": agent.limits})
    return root


def test_paid_source_review_revises_answer_without_replaying_new_experiments(source, tmp_path):
    analysis = """def answer(spec, observations, memory):
    row = observations[-1]
    active = any(obj['uuid'] == 105 and 'activated' in obj['name']
                 for obj in row['public_objects'])
    return {'status': 'supported' if active else 'inconclusive',
            'answer': {'activated': active}, 'evidence': [row['index']]}
"""
    provider = ScriptedProvider([("evaluate", {"question_id": "reference_extension_5",
        "path": "shared/analysis_reference_extension_5.py", "code": analysis})])
    agent, lab = study.review_source(source, tmp_path / "review", lambda _: provider, calls=1)
    saved = json.loads((tmp_path / "review/result.json").read_text())
    assert saved["review_status"] == "conclusion_revised"
    assert saved["new_calls"] == 1 and saved["new_actions"] == 0
    assert len(lab.actions) == len(agent.observations)
    question = next(q for q in agent.questions if q["id"] == "reference_extension_5")
    assert question["status"] == "supported" and question["answer"]["activated"]
    knowledge, archive = study.source_knowledge(source, agent, 82)
    assert archive["review_origin"] == "operator_requested_source_analysis_review"
    assert "original_questions" in archive and "measurement_history" in archive
    assert not any(path.endswith("reference_recalibration.json") for path in knowledge["files"])
    assert "target_conditions" not in json.dumps(knowledge)


def test_review_denies_new_world_action_and_keeps_source_account(source, tmp_path):
    provider = ScriptedProvider([("query", {"actions": [{"action": "PICKUP", "arg1": 93}]})])
    agent, _ = study.review_source(source, tmp_path / "review", lambda _: provider, calls=1)
    saved = json.loads((tmp_path / "review/result.json").read_text())
    assert saved["review_status"] == "review_call_limit" and saved["new_actions"] == 0
    assert "no world execution" in saved["review_events"][-1]["feedback"]["error"]
    assert agent.costs["model_calls"] == saved["costs_before"]["model_calls"] + 1


def test_review_restores_flat_episode_into_its_own_laboratory(source, tmp_path):
    flat_source = tmp_path / "flat_source"
    shutil.copytree(source, flat_source)
    previous = json.loads((flat_source / "result.json").read_text())["agent"]
    study.write_json(flat_source / "result.json", previous)
    lab = study.AliasedSignalLab()
    provider = ScriptedProvider([("finish", {"lesson": "Keep the unresolved conclusion"})])
    reviewed, restored = study.review_source(flat_source, tmp_path / "review", lambda _: provider,
        calls=1, laboratory_factory=lambda: lab)
    result = json.loads((tmp_path / "review/result.json").read_text())
    assert restored is lab
    assert result["replay"]["status"] == "matched" and result["new_actions"] == 0
    assert reviewed.observations == json.loads((source / "agent/laboratory_observations.json").read_text())
    assert reviewed.costs["model_calls"] == previous["costs"]["model_calls"] + 1
    assert (reviewed.workspace / study.METHOD_PATH).read_text() == (source / "agent" / study.METHOD_PATH).read_text()


def test_unpublished_author_never_opens_transfer_and_practice_uses_no_model(source, tmp_path, monkeypatch):
    # A failed author is a retained negative outcome, not a transferable method.
    original_knowledge = study.source_knowledge

    def failed_draft(*args):
        knowledge, archive = original_knowledge(*args)
        knowledge['files'][study.METHOD_PATH] = (
            "def select_actions(spec, observations, memory):\n    return {'stop': 'Draft needs research'}\n")
        return knowledge, archive

    monkeypatch.setattr(study, "source_knowledge", failed_draft)
    monkeypatch.setattr(study, "run_world_method_author", lambda agent, **kw: "model_call_budget_exhausted")
    monkeypatch.setattr(study, "transfer", lambda *a, **kw: pytest.fail("Unpublished draft opened targets"))
    result = study.run(source, tmp_path / "study", provider_factory=lambda _: ScriptedProvider([
        ("finish", {"lesson": "Mock review ends without a new conclusion"})]), calls=1, review_calls=1)
    assert result["transfer"] == "not_run_unpublished_draft"
    trials = list((tmp_path / "study/author/trials").glob("trial_*/result.json"))
    assert len(trials) == 2
    for path in trials:
        paid = json.loads(path.read_text())
        assert paid["costs"]["model_calls"] == 0
    protocol = json.loads((tmp_path / "study/protocol.json").read_text())
    assert protocol["target_conditions"] == study.CASES
    assert not (tmp_path / "study/method_bundle.json").exists()


def test_prespecified_world_changes_do_not_return_control_targets_as_measurements():
    good = study.TransferSignalLab("parameter_low")
    bad_sensor = study.TransferSignalLab("uninformative_sensor")
    changed = study.TransferSignalLab("changed_mechanism")
    assert good.required == bad_sensor.required
    assert bad_sensor.signal == bad_sensor.density and good.signal != good.density
    assert changed.signal == good.signal and changed.required != good.required
    assert max(changed.required.values()) < 10000
    reading = good.act({"action": "USE", "arg1": 93, "arg2": 202})
    assert "signal magnitude = 2.0 units" in reading["observation"]["lastActionMessage"]


def test_transfer_activation_updates_task_score_inside_dialog():
    lab = study.TransferSignalLab("parameter_low")
    lab.installed.update((4, 5))
    lab.frequencies[4] = round(lab.required[4])
    lab.frequencies[5] = round(lab.required[5]) - 1
    lab.act({"action": "TALK", "arg1": 105})
    assert not lab.score()["completedSuccessfully"]

    activated = lab.act({"chosen_dialog_option_int": 5})
    assert lab.in_dialog
    assert activated["observation"]["taskProgress"][0]["completed"]
    assert lab.score()["completedSuccessfully"]

    inactive = lab.act({"chosen_dialog_option_int": 9})
    assert not inactive["observation"]["taskProgress"][0]["completed"]
    assert not lab.score()["completedSuccessfully"]


@pytest.mark.parametrize("live", [False, True])
def test_transfer_uses_live_inquiry_only_when_a_provider_is_supplied(tmp_path, live):
    source = """def select_actions(spec, observations, memory):
    if not memory['model_available']:
        assert memory['limits']['max_model_calls'] == 0
        return {'stop': 'Frozen execution has no live researcher'}
    assert memory['limits']['max_model_calls'] == 3
    if memory['llm_inquiries'].get('extension', {}).get('invocations', 0):
        return {'stop': 'Research returned without a new method'}
    return {'action': 'run', 'arguments': {'operator': 'inquire',
        'max_model_calls': 3, 'question': {'id': 'extension', 'parent_id': 'task',
        'question': 'Review the existing evidence and return'}}}
"""
    package = {"files": {study.METHOD_PATH: source}, "methods": [{"id": "existing",
        "path": study.METHOD_PATH, "parameters": {}, "requires": [], "max_steps": 256}]}
    provider = ScriptedProvider([("finish", {"lesson": "No new evidence was acquired"})])
    factory_paths, checkpoints = [], []

    def factory(workspace):
        factory_paths.append(workspace)
        return provider

    limits = {**LIMITS, "max_model_calls": 3}
    row = study.run_transfer_case("parameter_low", "test", package, tmp_path, limits,
        provider_factory=factory if live else None, on_checkpoint=checkpoints.append)

    assert row["status"] == "method_stopped" and row["source_unchanged"]
    assert row["execution_kind"] == ("live_research" if live else "frozen_source")
    assert row["costs"]["model_calls"] == len(provider.requests) == int(live)
    assert row["costs"]["environment_steps"] == 0
    assert limits["max_model_calls"] == 3
    if live:
        assert factory_paths == [tmp_path / "parameter_low/test/agent"]
        assert provider.requests[0]["llm_inquiry"]["remaining_model_calls"] == 3
        assert checkpoints[-1]["costs"]["model_calls"] == 1
    else:
        assert not factory_paths and not checkpoints


def test_reuse_paid_review_makes_no_new_model_call_or_world_action(source, tmp_path):
    old, _ = study.review_source(source, tmp_path / "first", lambda _: ScriptedProvider([
        ("finish", {"lesson": "Previously paid review"})]), calls=1)
    reviewed, _ = study.review_source(source, tmp_path / "reused", lambda _: pytest.fail("Review model recreated"),
        reuse_result=tmp_path / "first/result.json")
    result = json.loads((tmp_path / "reused/result.json").read_text())
    assert result["review_status"] == "reused_paid_review"
    assert result["new_calls"] == result["new_actions"] == 0
    assert reviewed.costs["model_calls"] == old.costs["model_calls"]
    assert reviewed.observations == old.observations and reviewed.questions == old.questions


def test_new_seed_keeps_prior_author_as_experience_without_inheriting_support(source, tmp_path, monkeypatch):
    study.review_source(source, tmp_path / "review", lambda _: ScriptedProvider([
        ("finish", {"lesson": "Previously paid review"})]), calls=1)
    stop = "def select_actions(spec, observations, memory):\n    return {'stop': 'Research needed'}\n"
    seed = tmp_path / "seed.json"
    study.write_json(seed, {"files": {study.METHOD_PATH: stop, "shared/reference_recalibration.json": "OLD STATE"},
        "methods": [{"id": "old", "supported": True}], "fresh_confirmation": {"passed": True}})
    previous_workspace = tmp_path / "previous/author"
    (previous_workspace / "shared").mkdir(parents=True)
    (previous_workspace / study.METHOD_PATH).write_text("ESTIMATORS = ({'id': 'previous_scientific_candidate'},)\n" + stop)
    (previous_workspace / "research_notes.md").write_text("Preserve this earlier failed candidate and its evidence.")
    previous = tmp_path / "previous/result.json"
    study.write_json(previous, {"workspace": str(previous_workspace), "status": "model_call_budget_exhausted",
        "costs": {"model_calls": 24}, "events": [], "practice_trials": []})

    def author(learner, **kwargs):
        assert learner.files[study.METHOD_PATH] == stop
        assert "shared/reference_recalibration.json" not in learner.knowledge["files"]
        assert learner.published_source is None and learner.costs["model_calls"] == 0
        assert len(learner.practice_trials) == 2
        assert not any(t["fresh_confirmation"]["passed"] for t in learner.practice_trials)
        assert "Preserve this earlier failed candidate" in kwargs["source_experiences"][-1]["notes"]
        archive = json.loads(learner.knowledge["files"]["shared/previous_author_experience.json"])
        assert archive["candidates"] == [{"id": "previous_scientific_candidate"}]
        kwargs["on_checkpoint"]("model_call_budget_exhausted")
        return "model_call_budget_exhausted"

    monkeypatch.setattr(study, "run_world_method_author", author)
    monkeypatch.setattr(study, "transfer", lambda *a: pytest.fail("Unpublished draft opened targets"))
    result = study.run(source, tmp_path / "new", provider_factory=lambda _: ScriptedProvider([]), calls=1,
        initial_bundle=seed, review_result=tmp_path / "review/result.json", previous_author=previous)
    assert result["transfer"] == "not_run_unpublished_draft"
    protocol = json.loads((tmp_path / "new/protocol.json").read_text())
    assert protocol["review_calls"] == 0
    assert protocol["prior_work"]["source_review"]["model_calls"] == 1
    assert protocol["prior_work"]["previous_author"]["model_calls"] == 24
    assert "phase's starting source" in protocol["transfer_baseline"]
