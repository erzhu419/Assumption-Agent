from copy import deepcopy
import json

from scripts import run_response_assignment_comparison as comparison
from test_discoveryworld_agent import LIMITS


def test_assignment_reuses_instruction_without_old_inline_measurements(tmp_path):
    history = tmp_path / "history.json"
    original = "Submit an executable hypothesis. Current measured table: {\"old\": 178}"
    history.write_text(json.dumps({"llm_inquiry": {"question": {"question": original}}}))
    texts, raw = comparison.assignments(history)
    assert raw == original
    assert "old" not in texts["historical_assignment"]
    assert texts["historical_assignment"].startswith("Submit an executable hypothesis.")
    assert all("shared/reference_recalibration.json" in text for text in texts.values())


def test_same_paid_state_is_restored_and_only_assignment_text_changes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    config = {"intercept": 40.2, "gain": 1., "power": 3, "sensor": "informative"}
    monkeypatch.setattr(comparison.matched.study, "CASES", {"changed_mechanism": config})
    lab = comparison.matched.study.TransferSignalLab("changed_mechanism")
    prior = comparison.DiscoveryWorldAgent(None, source / "agent", arm="hegel", limits=LIMITS)
    prior.lab = lab
    prior._question({"id": "task", "question": "Calibrate targets"})
    prior._execute("write", {"path": comparison.matched.study.METHOD_PATH,
        "code": "ESTIMATORS = [{'id': 'old', 'basis': []}]\ndef select_actions(memory):\n    return []\n"}, lab)
    prior._execute("write", {"path": "shared/prior.json", "code": '{"earlier_evidence": true}'}, lab)
    prior.observations = [{"index": 0, **lab.act({"action": "TALK", "arg1": 101})}]
    prior.events = [{"action": "past", "feedback": {"old": True}}]
    prior.costs.update(model_calls=24, environment_steps=1, python_execution_seconds=12,
                       input_tokens=120, output_tokens=20, research_compute_units=5)
    (prior.workspace / "research_notes.md").write_text("Existing failed explanation, keep it.")
    prior.status = "model_call_budget_exhausted"
    prior._save_state()
    saved = prior.result()
    (source / "result.json").write_text(json.dumps(saved))

    restored = []
    for treatment in comparison.TREATMENTS:
        agent, world, context = comparison.restore(source, tmp_path / treatment, treatment,
            config=config, base_limits=LIMITS)
        assert agent.provider is None and agent.status == "running"
        assert agent.observations == prior.observations
        assert agent.events == prior.events
        assert agent.costs == saved["costs"] == context["costs_before"]
        assert world.environment_steps == 1 and world.frequencies == lab.frequencies
        assert (agent.workspace / "research_notes.md").read_text() == "Existing failed explanation, keep it."
        assert (agent.workspace / "shared/prior.json").read_text() == '{"earlier_evidence": true}'
        assert agent.limits["max_model_calls"] == 48
        assert agent.limits["environment_steps"] == 513
        assert agent.limits["python_seconds_per_task"] == 612
        request = comparison.prepared_request(agent, context)
        assert agent.costs["model_calls"] == 24
        assert request["llm_inquiry"]["remaining_model_calls"] == 20
        assert request["llm_inquiry"]["depth"] == 1
        # The same question text is naturally repeated in the root and method view.
        restored.append(json.loads(json.dumps(request).replace(treatment, "ASSIGNMENT")))
    assert restored[0] == restored[1]
    assert json.loads((source / "result.json").read_text())["status"] == "model_call_budget_exhausted"


def test_brief_does_not_recount_old_confirmations_or_old_model_decisions(monkeypatch):
    old = {"physically_confirmed": True, "actual_positive_evidence": [{"index": 100}]}
    new = {"physically_confirmed": True, "actual_positive_evidence": [{"index": 185}]}
    monkeypatch.setattr(comparison.matched, "point_confirmations", lambda result: [old, new])

    def trace(result):
        assert result["events"] == [{"inquiry_question_id": "new"}]
        return {"inquiries": [{"question_id": "old"}, {"question_id": "new"}], "decisions": []}

    monkeypatch.setattr(comparison.matched, "scientific_trace", trace)
    monkeypatch.setattr(comparison, "physical_responses", lambda result, points: [])
    result = {"treatment": "current_assignment", "repetition": 1, "status": "completed",
        "score": {"completedSuccessfully": True}, "prefix_observations": 184,
        "observations": list(range(186)), "events": [{"inquiry_question_id": "old"}],
        "new_events": [{"inquiry_question_id": "new"}], "declarations_before": [{"id": "old"}],
        "declarations_after": [{"id": "old"}, {"id": "new"}]}
    row = comparison.brief(result)
    assert row["new_observations"] == 2 and row["point_confirmations"] == [new]
    assert row["added_or_revised_declarations"] == [{"id": "new"}]
    assert row["scientific_trace"]["inquiries"] == [{"question_id": "new"}]


def test_terminal_activation_is_not_an_invented_supported_answer():
    from test_world_reference_hypothesis import ReferenceLab

    lab = ReferenceLab()
    lab.installed.add(4)
    lab.frequencies[4] = lab.required[4]
    old = {"index": 183, **lab.act({"action": "TALK", "arg1": 104})}
    repeated = {"index": 184, **lab.act({"action": "TALK", "arg1": 104})}
    lab.installed.add(5)
    lab.frequencies[5] = lab.required[5]
    new = {"index": 185, **lab.act({"action": "TALK", "arg1": 105})}
    rows = comparison.physical_responses({"observations": [old, repeated, new],
                                         "prefix_observations": 184}, [])
    assert [(r["crystal"], r["new_target_completion"], r["formal_confirmation_recorded"])
            for r in rows] == [(4, False, False), (5, True, False)]
