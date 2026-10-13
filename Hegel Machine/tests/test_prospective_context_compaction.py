"""Authored comparison practice returns stay readable within the provider limit."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition import code_provider_context


@pytest.mark.parametrize("automatic_practice", [True, False])
def test_comparison_practice_keeps_repair_feedback_and_retrievable_forecasts(
        tmp_path, automatic_practice):
    source = {"model": "candidate.py", "files": {"candidate.py": "# fitted model\n" * 4000}}
    inputs = [{"features": {"u": index / 72},
               "history": [{"u": earlier / 72} for earlier in range(index)]}
              for index in range(72)]
    forecasts = [{"model": f"candidate_{index}.py", "source": source,
                  "prediction_inputs": inputs, "predictions": [row / 72 for row in range(72)],
                  "returncode": int(index == 2),
                  "stderr": "ValueError: candidate fit failed" if index == 2 else ""}
                 for index in range(3)]
    proposal = {"action": "run", "arguments": {"operator": "compare_experiments",
                "models": [item["model"] for item in forecasts],
                "experiments": [{"id": "probe", "actions": [{"kind": "trajectory",
                                 "sequence": [{"u": 0.0}, {"u": 1.0}]}]}]}}
    comparison = {"index": 0, "forecasts": forecasts,
                  "prior_fit": [{"model": "candidate_0.py", "mse": 1e-30}],
                  "candidate_contexts": [{"source_model": "candidate_0.py", "model": "frozen.py"}],
                  "selected_experiment": {"id": "probe", "predicted_separation": .4},
                  "query_arguments": {"revision": {"comparison_index": 0}},
                  "status": "compared", "diagnosis": "discriminating_experiment_available"}
    practice = {"directory": "trials/trial_036", "status": "step_budget_exhausted",
                "steps": [{"step": index, "proposal": proposal,
                           "feedback": {"comparison": comparison, "program_action": proposal}}
                          for index in range(32)],
                "experimental_design": {"candidate_probes": 1, "purchased": 0}}
    feedback = ({"written": "shared/method.py", "edit_practice": practice,
                 "experimental_design": {"status": "incomplete"}}
                if automatic_practice else practice)
    request = {"events": [{"action": "write" if automatic_practice else "run",
                           "arguments": {} if automatic_practice else {"operator": "practice"},
                           "feedback": feedback}]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    stored = json.loads((tmp_path / "research_context.json").read_text())
    assert stored == original == request
    assert len(json.dumps(request)) > 1_048_576
    assert len(json.dumps(context)) < 1_048_576
    compact = context["events"][0]["feedback"]
    if automatic_practice:
        assert compact["experimental_design"] == feedback["experimental_design"]
        compact = compact["edit_practice"]
    assert compact["full_trace_file"] == "trials/trial_036/result.json"
    assert compact["status"] == practice["status"]
    assert compact["experimental_design"] == practice["experimental_design"]
    assert [step["step"] for step in compact["steps"]] == list(range(32))
    for step in compact["steps"]:
        assert step["proposal"] == step["feedback"]["program_action"] == proposal
        result = step["feedback"]["comparison"]
        for key in ("prior_fit", "candidate_contexts", "selected_experiment", "query_arguments",
                    "status", "diagnosis"):
            assert result[key] == comparison[key]
        for expected, actual in zip(forecasts, result["forecasts"]):
            assert actual["stderr"] == expected["stderr"]
            assert actual["returncode"] == expected["returncode"]
            for field in ("source", "prediction_inputs", "predictions"):
                reference = actual[field]
                assert reference["full_content_file"] == "research_context.json"
                original_value = stored
                for key in reference["json_path"]:
                    original_value = original_value[key]
                assert original_value == expected[field]
            assert actual["predictions"]["count"] == 72
            assert actual["predictions"]["preview"] == [expected["predictions"][i] for i in (0, 1, 70, 71)]


def test_two_large_practices_bound_the_total_request_and_keep_latest_feedback(tmp_path, monkeypatch):
    actions = [{"kind": "trajectory", "sequence": [
        {"u": (-1) ** index * (index + .123456789) / 73} for index in range(72)]} for _ in range(3)]
    proposal = {"action": "run", "arguments": {"operator": "compare_experiments",
        "models": ["prior.py", "polynomial.py", "latent.py"],
        "experiments": [{"id": "reversal_and_holds", "actions": actions}]}}
    comparison = {"status": "compared", "selected_experiment": {
        "id": "reversal_and_holds", "predicted_separation": .16, "actions": actions},
        "query_arguments": {"actions": actions, "revision": {"comparison_index": 0}}}
    practice = {"case": "design_case_005", "directory": "trials/trial_038",
        "status": "method_step_limit", "costs": {"observation_cost": 72},
        "fresh_confirmation": None, "steps": [
            {"step": index, "path": "shared/consolidated_method.py", "proposal": proposal,
             "feedback": {"comparison": comparison, "program_action": proposal}}
            for index in range(32)]}
    latest = deepcopy(practice)
    latest['directory'] = 'trials/trial_039'
    latest['steps'][-1]['feedback']['error'] = 'Actual latest practice error'
    request = {"task_interface": "Choose an ordinary action using retained observations.",
        "costs": {"model_calls": 3}, "limits": {"max_model_calls": 93}, "events": [
            {"action": "run", "arguments": {"operator": "practice"}, "feedback": practice},
            {"action": "write", "arguments": {"path": "shared/consolidated_method.py",
                                                  "code": "unchanged authored source"},
             "feedback": {"written": "shared/consolidated_method.py", "edit_practice": latest}}]}
    with monkeypatch.context() as patch:
        patch.setattr(code_provider_context, '_bound_provider_context', lambda context: context)
        before = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    assert len(json.dumps(before)) > 1_048_576
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    assert len(json.dumps(context)) <= 1_000_000
    stored = json.loads((tmp_path / 'research_context.json').read_text())
    assert stored == request
    assert context['events'][-1] == before['events'][-1]
    assert context['costs'] == request['costs'] and context['task_interface'] == request['task_interface']
    old_feedback = context['events'][0]['feedback']
    assert old_feedback['status'] == practice['status']
    assert [step['step'] for step in old_feedback['steps']] == list(range(32))
    for reference in old_feedback['steps']:
        value = stored
        for key in reference['json_path']:
            value = value[key]
        assert value == practice['steps'][reference['step']]


def test_global_context_bound_applies_to_accumulated_sections_not_only_practices(tmp_path):
    request = {'task_interface': 'Read the named files to inspect the complete evidence.',
        'files': {f'shared/model_{index:03d}.py': '# paid fitted coefficients\n' * 300
                  for index in range(160)},
        'events': [{'action': 'run', 'arguments': {'code': 'inspect()'},
                    'feedback': {'returncode': 1, 'stderr': 'actual diagnostic retained'}}]}
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    assert len(json.dumps(context)) <= 1_000_000
    assert context['events'][-1]['feedback'] == request['events'][-1]['feedback']
    assert json.loads((tmp_path / 'research_context.json').read_text()) == request
    reference = context['files']
    assert reference['full_content_file'] == 'research_context.json'
    assert reference['json_path'] == ['files']
