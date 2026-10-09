"""Transfer preparation separates public source experience from target evidence."""

import json

import pytest

from scripts import prepare_discoveryworld_transfer_method as author
from scripts import run_discoveryworld_method_transfer as transfer
from test_discoveryworld_agent import LIMITS
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_world_method_learning import SOLVE
from test_world_method_multisource import SeedLab


@pytest.fixture
def successful_source(tmp_path):
    root = tmp_path / "source_study"
    source = root / "dw_04_normal_s0" / "hegel"
    workspace = source / "agent"
    workspace.mkdir(parents=True)
    observations = []
    for index, action in enumerate([
        {"action": "USE", "arg1": 5, "arg2": 7},
        {"action": "TALK", "arg1": 9},
        {"action": "USE", "arg1": 5, "arg2": 8},
        {"chosen_dialog_option_int": 2},
    ]):
        observations.append({
            "index": index, "action": action, "feedback": {"success": True},
            "observation": {"lastActionMessage": f"Public action {index}",
                            "taskProgress": [{"description": "Public scientific task"}],
                            "inventoryObjects": [{"uuid": 5, "name": "measurement device"}]},
        })
    result = {
        "task_id": "dw_04_normal_s0",
        "score": {"completedSuccessfully": True, "private_scorecard": "PRIVATE_GOLD_RULE"},
        "score_checkpoints": [{"private_scorecard": "PRIVATE_CHECKPOINT"}],
        "costs": {"model_calls": 4, "input_tokens": 100, "environment_steps": 4},
        "prior_costs": {"model_calls": 12, "input_tokens": 300, "environment_steps": 2},
        "prefix_event_count": 1, "prefix_environment_steps": 2,
        "events": [
            {"action": "query", "arguments": {"notes": "PRE_CONTINUATION_EVENT"},
             "feedback": {"observation_indices": [0, 1]}, "usage": {"input_tokens": 90}},
            {"action": "run", "arguments": {"code": "print('public calculation')"},
             "feedback": {"stdout": "public calculation"}, "usage": {"input_tokens": 50}},
            {"action": "query", "arguments": {"actions": [{"chosen_dialog_option_int": 2}]},
             "feedback": {"observation_indices": [2, 3]}, "usage": {"input_tokens": 50}},
        ],
    }
    context = {"capabilities": {"actions": ["USE", "TALK"]},
               "current_observation": {"lastActionMessage": "STALE_PRE_ACTION_SNAPSHOT"},
               "private_evaluation": "PRIVATE_CONTEXT_SCORECARD"}
    plan = {"model": {"backend": "codex_cli", "requested_alias": "gpt-6-luna",
                      "reasoning_effort": "medium"},
            "limits": {**LIMITS, "max_model_calls": 24, "provider_request_timeout_seconds": 900},
            "runtime": "/source/runtime"}
    for path, value in [(source / "result.json", result),
                        (workspace / "laboratory_observations.json", observations),
                        (workspace / "research_context.json", context),
                        (root / "protocol.json", plan)]:
        path.write_text(json.dumps(value))
    return source, result, observations, context, plan


def test_source_export_contains_public_experience_and_correct_continuation_only(successful_source):
    source, result, observations, context, _ = successful_source
    archive = author.source_archive(source)
    rendered = json.dumps(archive)
    assert "PRIVATE_" not in rendered
    assert "costs" not in archive and "score" not in archive
    assert all("usage" not in event for event in archive["events"])
    assert archive["public_spec"] == {
        "capabilities": context["capabilities"], "current_observation": observations[-1]["observation"]}
    assert [event["action"] for event in archive["events"]] == ["run", "query"]
    assert "PRE_CONTINUATION_EVENT" not in rendered
    assert "STALE_PRE_ACTION_SNAPSHOT" not in rendered
    assert [record["index"] for record in archive["continuation_observations"]] == [2, 3]
    # Earlier measurements remain available as historical inputs, but are not
    # relabeled as actions executed in this final continuation.
    assert [record["index"] for record in archive["measurement_history"]] == [0, 2]
    assert archive["events"][0]["feedback"] == result["events"][1]["feedback"]


def test_source_export_requires_a_successful_source_episode(successful_source):
    source, result, *_ = successful_source
    result["score"]["completedSuccessfully"] = False
    (source / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="actually successful source"):
        author.source_archive(source)


def test_ordinary_successful_episode_exports_without_continuation_prefix(successful_source):
    source, result, observations, _, _ = successful_source
    del result["prefix_event_count"]
    del result["prefix_environment_steps"]
    del result["prior_costs"]
    result["research"] = {"questions": [{"id": "old_source_answer", "status": "supported",
                                        "answer": "OLD_SOURCE_SUPPORT", "evidence": [0]}]}
    (source / "result.json").write_text(json.dumps(result))

    archive = author.source_archive(source)

    assert len(archive["events"]) == len(result["events"])
    assert [record["index"] for record in archive["continuation_observations"]] == [
        record["index"] for record in observations]
    assert archive["record_kind"] == "source_experience_not_consolidated_program_execution"
    assert "OLD_SOURCE_SUPPORT" not in json.dumps(archive)
    assert "PRIVATE_" not in json.dumps(archive)


def test_additional_source_keeps_actual_task_costs_without_importing_old_support(successful_source):
    source, result, _, _, plan = successful_source
    task = {"id": "dw_04_normal_s3", "random_seed": 3,
            "difficulty": "Normal", "split": "evaluation"}
    result["task_id"] = task["id"]
    result["research"] = {"questions": [{"id": "old_answer", "status": "supported",
                                        "answer": "OLD_SOURCE_SUPPORT", "evidence": [2]}]}
    del result["prefix_event_count"]
    del result["prefix_environment_steps"]
    plan["tasks"] = [{**task, "id": "unused", "random_seed": 99}, task]
    (source / "result.json").write_text(json.dumps(result))
    (source.parents[1] / "protocol.json").write_text(json.dumps(plan))

    selected, archive, provenance = author.additional_source(source)

    assert selected == task
    assert provenance["task"] == task and provenance["costs"] == result["costs"]
    assert provenance["source"] == str(source.resolve())
    assert "development" in provenance["scope"]
    assert "OLD_SOURCE_SUPPORT" not in json.dumps(archive)
    assert archive["record_kind"] == "source_experience_not_consolidated_program_execution"

    result["score"]["completedSuccessfully"] = False
    (source / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="actually successful source"):
        author.additional_source(source)


def test_flat_successful_child_source_imports_current_task_and_public_discovery(
        successful_source, tmp_path):
    source, result, observations, context, plan = successful_source
    flat_source = source.rename(tmp_path / "continued_child_research")
    task = {"id": "dw_04_normal_s4", "random_seed": 4,
            "difficulty": "Normal", "split": "evaluation"}
    result["task_id"] = task["id"]
    result.pop("prefix_event_count")
    result.pop("prefix_environment_steps")
    result["replayed_environment_steps"] = 2
    result["events"] = [{
        "action": "run", "inquiry_question_id": "new_observable",
        "arguments": {"code": "print('current reference fit predicts 2228.36')",
                      "research_question": "new_observable",
                      "notes": "Fit a new observable from current reference measurements"},
        "feedback": {"returncode": 0, "stdout": "current reference fit predicts 2228.36"},
        "usage": {"input_tokens": 50},
    }]
    result["research"] = {"questions": [{"id": "new_observable", "status": "supported",
        "answer": "OLD_SOURCE_SUPPORT", "evidence": [2]}]}
    context["research_notes"] = "The earlier observable failed; the measured new relation worked."
    plan.pop("tasks", None)
    plan["task"] = task
    (flat_source / "protocol.json").write_text(json.dumps(plan))
    (flat_source / "result.json").write_text(json.dumps(result))
    (flat_source / "agent/research_context.json").write_text(json.dumps(context))

    selected, archive, provenance = author.additional_source(flat_source)

    assert selected == task
    assert provenance["task"] == task and provenance["costs"] == result["costs"]
    assert provenance["source"] == str(flat_source.resolve())
    assert "development" in provenance["scope"]
    assert archive["notes"] == context["research_notes"]
    assert archive["events"] == [{key: result["events"][0][key]
                                  for key in ("action", "arguments", "feedback")}]
    assert [record["index"] for record in archive["continuation_observations"]] == [2, 3]
    assert [record["index"] for record in archive["measurement_history"]] == [0, 2]
    assert archive["public_spec"]["current_observation"] == observations[-1]["observation"]
    assert archive["record_kind"] == "source_experience_not_consolidated_program_execution"
    assert "OLD_SOURCE_SUPPORT" not in json.dumps(archive)
    assert "PRIVATE_" not in json.dumps(archive)
    assert "research" not in archive and "usage" not in archive["events"][0]


@pytest.mark.parametrize("protocol_layout", ["flat", "nested"])
def test_additional_source_rejects_protocol_for_a_different_task(
        successful_source, protocol_layout):
    source, _, _, _, plan = successful_source
    wrong_task = {"id": "dw_04_normal_s4", "random_seed": 4,
                  "difficulty": "Normal", "split": "evaluation"}
    if protocol_layout == "flat":
        plan["task"] = wrong_task
        protocol_path = source / "protocol.json"
    else:
        plan["tasks"] = [wrong_task]
        protocol_path = source.parents[1] / "protocol.json"
    protocol_path.write_text(json.dumps(plan))

    with pytest.raises(ValueError, match="actual task"):
        author.additional_source(source)


def test_replayed_prefix_is_history_not_new_child_actions(successful_source):
    source, result, _, _, _ = successful_source
    result.pop("prefix_environment_steps")
    result.pop("prefix_event_count")
    result["replayed_environment_steps"] = 2
    # Child results already contain only the new decisions. The world replay
    # count must slice observations, never this separately recorded event list.
    result["events"] = result["events"][1:]
    (source / "result.json").write_text(json.dumps(result))

    archive = author.source_archive(source)

    assert [event["action"] for event in archive["events"]] == ["run", "query"]
    assert [record["index"] for record in archive["continuation_observations"]] == [2, 3]
    assert [record["index"] for record in archive["measurement_history"]] == [0, 2]


def test_transfer_plan_retains_registered_evaluation_tasks_and_matched_source_costs(
        successful_source, tmp_path, monkeypatch):
    source, result, _, _, source_plan = successful_source
    consolidation_costs = {"model_calls": 3, "input_tokens": 75}
    bundle_path = tmp_path / "method_bundle.json"
    bundle_path.write_text(json.dumps({"files": {}, "methods": [], "source": str(source),
                                      "consolidation_costs": consolidation_costs,
                                      "validation": "Registered, not yet executed on a target"}))
    monkeypatch.setattr(transfer, "verify_codex_subscription", lambda: "test_subscription")
    plan = transfer.make_transfer_plan(bundle_path, calls=64, workers=4)
    registered = json.loads(transfer.SELECTION.read_text())["tasks"]
    target_ids = ["dw_04_normal_s3", "dw_04_normal_s4"]
    assert plan["tasks"] == [next(task for task in registered if task["id"] == task_id)
                             for task_id in target_ids]
    assert plan["split"] == "evaluation"
    assert all(task["split"] == "evaluation" for task in plan["tasks"])
    assert plan["arms"] == ["code_archive", "hegel"]
    assert plan["jobs"] == [
        {"task_id": target_ids[0], "arm": "code_archive"},
        {"task_id": target_ids[0], "arm": "hegel"},
        {"task_id": target_ids[1], "arm": "hegel"},
        {"task_id": target_ids[1], "arm": "code_archive"},
    ]
    assert plan["limits"] == {**source_plan["limits"], "max_model_calls": 64}
    assert plan["model"] == source_plan["model"]
    assert plan["runtime"] == source_plan["runtime"] and plan["workers"] == 4
    assert plan["method_bundle"] == str(bundle_path.resolve())
    assert plan["source_prior_costs"] == result["prior_costs"]
    assert plan["source_last_costs"] == result["costs"]
    assert plan["consolidation_costs"] == consolidation_costs
    assert plan["auth_mode"] == "test_subscription"


@pytest.mark.parametrize("development_id", ["dw_04_normal_s0", "dw_04_normal_s1"])
def test_transfer_cannot_relabel_a_development_world_as_frozen_evaluation(
        successful_source, tmp_path, monkeypatch, development_id):
    source, *_ = successful_source
    bundle_path = tmp_path / "method_bundle.json"
    bundle_path.write_text(json.dumps({"files": {}, "methods": [], "source": str(source),
        "development_task_ids": ["dw_04_normal_s1"], "consolidation_costs": {},
        "validation": "Both source worlds practiced"}))
    monkeypatch.setattr(transfer, "verify_codex_subscription", lambda: "test_subscription")

    with pytest.raises(ValueError, match="development"):
        transfer.make_transfer_plan(bundle_path, target_ids=[development_id, "dw_04_normal_s3"])


def frozen_bundle(tmp_path, source, name, program, *, development=("dw_04_normal_s0",)):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"files": {METHOD_PATH: program},
        "methods": [{"id": "source_method", "path": METHOD_PATH,
                     "parameters": {"gain": 2}, "requires": []}],
        "source": str(source), "development_task_ids": list(development),
        "consolidation_costs": {}, "validation": "Source program practiced"}))
    return path


def no_subscription():
    raise AssertionError("Frozen execution must not require a model subscription")


def test_frozen_transfer_executes_both_programs_with_registered_parameters_and_no_model(
        successful_source, tmp_path, monkeypatch):
    source, _, _, _, plan = successful_source
    measured = SOLVE.replace("2 * reading", "memory['program_arguments']['parameters']['gain'] * reading")
    current = frozen_bundle(tmp_path, source, "current", measured)
    reference = frozen_bundle(tmp_path, source, "reference", SOLVE.replace("2 * reading", "6"))
    agents, labs = [], []
    original_agent = transfer.DiscoveryWorldAgent

    class TrackedAgent(original_agent):
        def __init__(self, provider, workspace, *, arm, limits):
            assert provider is None
            super().__init__(provider, workspace, arm=arm, limits=limits)
            agents.append(self)

    def laboratory(task, runtime, workspace):
        assert runtime == plan["runtime"]
        lab = SeedLab(task)
        labs.append(lab)
        return lab

    monkeypatch.setattr(transfer, "verify_codex_subscription", no_subscription)
    monkeypatch.setattr(transfer, "DiscoveryWorldAgent", TrackedAgent)
    monkeypatch.setattr(transfer, "DiscoveryWorldLab", laboratory)
    output = tmp_path / "frozen"

    rows = transfer.run_frozen_transfer(current, output, reference_bundle=reference)

    assert len(rows) == len(agents) == len(labs) == 4
    assert [row["completedSuccessfully"] for row in rows] == [True, False, True, False]
    assert [lab.actions[-1]["arg1"] for lab in labs] == [14, 6, 14, 6]
    assert all(agent.limits == {**plan["limits"], "max_model_calls": 0} for agent in agents)
    assert all(row["costs"]["model_calls"] == 0 for row in rows)
    assert all(not agent.llm_inquiries for agent in agents)
    assert all((agent.workspace / METHOD_PATH).read_text() ==
               (measured if index % 2 == 0 else SOLVE.replace("2 * reading", "6"))
               for index, agent in enumerate(agents))
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["auth_mode"] == "not_used" and protocol["model"]["model_calls"] == 0
    assert json.loads((output / "brief.json").read_text()) == rows


def test_frozen_transfer_excludes_reference_program_development_tasks(
        successful_source, tmp_path, monkeypatch):
    source, *_ = successful_source
    current = frozen_bundle(tmp_path, source, "current", SOLVE)
    reference = frozen_bundle(tmp_path, source, "reference", SOLVE,
                              development=("dw_04_normal_s3",))
    monkeypatch.setattr(transfer, "verify_codex_subscription", no_subscription)

    def no_world(*args):
        raise AssertionError("Development overlap must be rejected before opening a target")

    monkeypatch.setattr(transfer, "DiscoveryWorldLab", no_world)

    with pytest.raises(ValueError, match="development"):
        transfer.run_frozen_transfer(current, tmp_path / "frozen", reference_bundle=reference)


def test_frozen_transfer_keeps_an_existing_output_unchanged(successful_source, tmp_path, monkeypatch):
    source, *_ = successful_source
    current = frozen_bundle(tmp_path, source, "current", SOLVE)
    output = tmp_path / "frozen"
    output.mkdir()
    previous = {"status": "earlier result must remain"}
    (output / "protocol.json").write_text(json.dumps(previous))
    monkeypatch.setattr(transfer, "verify_codex_subscription", no_subscription)

    with pytest.raises(FileExistsError):
        transfer.run_frozen_transfer(current, output)

    assert json.loads((output / "protocol.json").read_text()) == previous
    assert sorted(path.name for path in output.iterdir()) == ["protocol.json"]
