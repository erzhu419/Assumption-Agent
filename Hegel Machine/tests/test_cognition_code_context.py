"""The real provider can inspect all paid evidence without resending it each turn."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.code_provider_context import prepare_code_context


def test_large_programs_are_retrievable_without_repeating_current_or_retained_bodies(tmp_path):
    source = "# learned coefficients\n" * 16000
    old = source + "# earlier version\n"
    request = {"files": {"shared/model.py": source},
               "session_knowledge": {"files": {"shared/model.py": source, "shared/old.py": old}}}
    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")
    stored = json.loads((tmp_path / "research_context.json").read_text())
    assert stored == request
    assert "shared/model.py" not in context["session_knowledge"]["files"]
    for descriptor, expected in ((context["files"]["shared/model.py"], source),
                                 (context["session_knowledge"]["files"]["shared/old.py"], old)):
        value = stored
        for key in descriptor["json_path"]:
            value = value[key]
        assert value == expected
        assert descriptor["characters"] == len(expected)
    assert len(json.dumps(context)) < len(json.dumps(request)) / 100


def test_context_preview_keeps_full_evidence_retrievable_and_latest_error_visible(tmp_path, monkeypatch):
    observations = [{'features': {'u': index / 100}, 'target': index / 10,
                     'trajectory_start': index % 9 == 0} for index in range(73)]
    program = 'def predict(row):\n    return row["features"]["u"]\n'
    events = [
        {'action': 'query', 'arguments': {'sequence': [{'u': row / 100} for row in range(64)]},
         'feedback': {'observations_file': 'observations.json', 'observation_cost': 64}},
        {'action': 'run', 'arguments': {'code': 'print("old diagnostic")'},
         'feedback': {'returncode': 0, 'stdout': 'old diagnostic\n' * 500, 'stderr': ''}},
        {'action': 'run', 'arguments': {'code': 'raise ValueError("refit needed")'},
         'feedback': {'returncode': 1, 'stdout': '', 'stderr': 'ValueError: refit needed'}},
    ]
    request = {'observations': observations, 'events': events,
               'files': {'shared/model.py': program, 'shared/revised.py': 'new explanation'},
               'session_knowledge': {'files': {'shared/model.py': program, 'shared/revised.py': 'old explanation'}, 'lessons': ['check applicability']}}
    original = deepcopy(request)
    transmitted = []

    def completion(command, payload, timeout, cwd):
        assert (Path(cwd) / 'code_agent_instructions.txt').read_text() == code_agent._INSTRUCTIONS
        disabled = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--disable"]
        enabled = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--enable"]
        assert "code_mode_host" not in disabled and "code_mode_host" in enabled
        assert "shell_tool" in disabled and "multi_agent" in disabled
        context = json.loads(payload)
        transmitted.append(context)
        stored = json.loads((Path(cwd) / context['evidence']['full_context_file']).read_text())
        assert stored == original
        assert len(context['observations']) == 12
        assert context['observations'][0]['row_index'] == 61
        assert context['evidence']['observation_count'] == 73
        assert context['observations'][-1]['trajectory_start']
        assert len(context['events']) == len(events)
        assert context['events'][0]['arguments']['sequence_rows'] == 64
        assert 'sequence' not in context['events'][0]['arguments']
        assert 'stdout' not in context['events'][1]['feedback']
        assert context['events'][-1]['feedback']['stderr'] == 'ValueError: refit needed'
        assert context['files']['shared/model.py'] == program
        assert context['session_knowledge']['shared_files'] == ['shared/model.py', 'shared/revised.py']
        assert context['session_knowledge']['files'] == {'shared/revised.py': 'old explanation'}
        assert context['session_knowledge']['lessons'] == ['check applicability']
        destination = Path(command[command.index('--output-last-message') + 1])
        destination.write_text(json.dumps({'action': 'run', 'arguments_json': '{"code":"print(1)"}'}))
        warning = {'type': 'item.completed', 'item': {'type': 'error', 'message': 'runtime diagnostic'}}
        completed = {'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 2}}
        return 0, json.dumps(warning) + '\n' + json.dumps(completed), '', False

    monkeypatch.setattr(code_agent, '_run', completion)
    reply = code_agent.CodexCodeProvider(tmp_path, executable='/test/codex').next_action(request)
    assert reply['action'] == 'run'
    assert reply['usage']['total_tokens'] == 12
    assert reply['usage']['runtime_errors'] == ['runtime diagnostic']
    assert request == original
    assert len(json.dumps(transmitted[0])) < len(json.dumps(original)) / 2


def test_task_interface_provider_instructions_defer_action_and_predictor_contracts(tmp_path, monkeypatch):
    interface = ("query {actions:[{kind:'trajectory',sequence:[{u:number}],channels:[]}]}\n"
                 "Predict from current controls and control history only. Use the stated memory condition.")
    request = {"task_interface": interface, "capabilities": {"controls": ["u"], "max_probe_length": 24},
               "observations": [], "events": []}

    def completion(command, payload, timeout, cwd):
        setting = next(item for item in command if item.startswith("model_instructions_file="))
        instructions_path = Path(json.loads(setting.split("=", 1)[1]))
        instructions = instructions_path.read_text()
        assert instructions == code_agent._TASK_INTERFACE_INSTRUCTIONS
        assert "task_interface" in instructions and "capabilities" in instructions
        assert "__target__" not in instructions
        assert "query: {sequence" not in instructions
        assert "measurement_views" not in instructions
        assert "risk_quantile" not in instructions
        context = json.loads(payload)
        assert context["task_interface"] == interface
        assert context["capabilities"] == request["capabilities"]
        assert json.loads((Path(cwd) / "research_context.json").read_text()) == request
        destination = Path(command[command.index("--output-last-message") + 1])
        destination.write_text(json.dumps({"action": "run", "arguments_json": '{"code":"print(1)"}'}))
        completed = {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}
        return 0, json.dumps(completed), "", False

    monkeypatch.setattr(code_agent, "_run", completion)
    reply = code_agent.CodexCodeProvider(tmp_path, executable="/test/codex").next_action(request)
    assert reply["action"] == "run"
    assert reply["usage"]["total_tokens"] == 12


@pytest.mark.parametrize("latest_stdout", ["", "saved revised model\n"])
def test_context_keeps_latest_code_and_earlier_concise_fit_results(tmp_path, latest_stdout):
    diagnostic = "0.5 5.9402\n0.8 4.86397\n0.9 4.69697\n"
    code = "from pathlib import Path\nPath('model.py').write_text('def predict(row): return 0.0\\n')"
    request = {"events": [
        {"action": "run", "arguments": {"code": "print('candidate fit results')"},
         "feedback": {"returncode": 0, "stdout": diagnostic, "stderr": "", "executed_path": "scratch_000.py"}},
        {"action": "run", "arguments": {"code": code},
         "feedback": {"returncode": 0, "stdout": latest_stdout, "stderr": "", "executed_path": "scratch_001.py"}},
    ]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test only")

    previous, latest = context["events"]
    assert previous["arguments"] == {"path": "scratch_000.py"}
    assert previous["feedback"]["stdout"] == diagnostic
    assert latest["arguments"] == {"path": "scratch_001.py", "code": code}
    assert latest["feedback"]["stdout"] == latest_stdout
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original


def test_world_evidence_metadata_does_not_become_an_empty_numerical_dataset(tmp_path):
    records = [{"index": index, "observation": {"lastActionMessage": f"Reading {index}"}}
               for index in range(318)]
    evidence_path = tmp_path / "laboratory_observations.json"
    evidence_path.write_text(json.dumps(records))
    (tmp_path / "observations.json").write_text("[]")
    evidence = {"observations_file": "laboratory_observations.json", "observation_count": len(records),
                "record_fields": {"message": "observation.lastActionMessage"}}
    request = {"task_interface": "Use public world observations", "evidence": evidence, "events": []}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    assert "observations" not in context
    assert context["evidence"] == {**evidence, "full_context_file": "research_context.json"}
    assert json.loads((tmp_path / context["evidence"]["observations_file"]).read_text()) == records
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original


def test_recent_event_window_uses_actual_executor_paths_without_renumbering(tmp_path):
    events = [{"action": "run", "arguments": {"code": f"print({index})"},
               "feedback": {"returncode": 0, "stdout": str(index),
                            "executed_path": f"scratch_{index:03d}.py"}}
              for index in range(549, 555)]
    request = {"events": events}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    for expected, actual in zip(events, context["events"]):
        assert actual["arguments"]["path"] == expected["feedback"]["executed_path"]
    assert "code" not in context["events"][0]["arguments"]
    assert context["events"][-1]["arguments"]["code"] == events[-1]["arguments"]["code"]
    assert request == original


def test_trajectory_preview_compacts_repeated_history_without_changing_paid_files(tmp_path):
    records, observations = [], []
    for trajectory in range(4):
        sequence = [{"u": index / 24} for index in range(24)]
        rows = [{"features": point, "target": index / 10, "history": sequence[:index]}
                for index, point in enumerate(sequence)]
        indices = list(range(trajectory * 24, (trajectory + 1) * 24))
        record = {"observation_index": indices[0], "row_indices": indices,
                  "action": {"kind": "trajectory", "sequence": sequence, "channels": ["meter"]},
                  "rows": rows, "measurements": [{"observation_index": index, "values": {"meter": index / 100}}
                                                for index in indices],
                  "observation_cost": 24, "measurement_cost": 72}
        records.append(record)
        observations.extend(rows)
    point_record = {"observation_index": 96, "action": {"kind": "response", "point": {"u": .2}}, "value": .4}
    request = {"observations": observations, "laboratory_observations": [*records, point_record],
               "events": [
                   {"action": "query", "arguments": {"actions": [record["action"] for record in records]},
                    "feedback": {"observations": records, "observation_cost": 96}},
                   {"action": "run", "arguments": {"code": "raise ValueError('old failure')"},
                    "feedback": {"returncode": 1, "stderr": "ValueError: old failure"}},
                   {"action": "finish", "arguments": {"model": "model.py"},
                    "feedback": {"error": "fresh confirmation failed; revise using these rows",
                                 "stderr": "latest diagnostic", "stdout": "latest result",
                                 "fresh_confirmation": {"passed": False, "observations": [records[-1]]}}}]}
    original = deepcopy(request)
    laboratory_file = tmp_path / "laboratory_observations.json"
    laboratory_file.write_text(json.dumps(request["laboratory_observations"]))
    file_before = laboratory_file.read_bytes()
    context = prepare_code_context(tmp_path, request, schema={}, instructions="test only")

    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert laboratory_file.read_bytes() == file_before
    assert context["evidence"]["laboratory_observations_file"] == "laboratory_observations.json"
    assert all("history" not in row and "history_length" in row for row in context["observations"])
    assert context["observations"][-1]["history_length"] == 23
    compact = context["laboratory_observations"][0]
    assert compact["row_count"] == compact["measurement_count"] == 24
    assert [row["row_index"] for row in compact["rows_preview"]] == [0, 1, 22, 23]
    assert [row["history_length"] for row in compact["rows_preview"]] == [0, 1, 22, 23]
    assert len(compact["measurements_preview"]) == 4
    assert compact["action"] == {"kind": "trajectory", "channels": ["meter"], "sequence_rows": 24, "controls": ["u"]}
    assert context["events"][0]["arguments"]["actions"][0] == compact["action"]
    assert context["events"][0]["feedback"]["observations"][0] == compact
    assert context["laboratory_observations"][-1] == point_record
    assert context["events"][1]["feedback"]["stderr"] == "ValueError: old failure"
    latest = context["events"][-1]["feedback"]
    assert latest["error"] == request["events"][-1]["feedback"]["error"]
    assert latest["stderr"] == "latest diagnostic" and latest["stdout"] == "latest result"
    assert latest["fresh_confirmation"]["observations"][0]["row_count"] == 24
    assert '"history":' not in json.dumps(context)
    assert len(json.dumps(context)) < len(json.dumps(original)) / 3


def test_world_large_program_proposal_remains_available_from_full_context(tmp_path):
    estimators = [{'id': f'candidate_{index}', 'basis': [{'op': 'feature', 'name': f'x{i}'}
                  for i in range(20)]} for index in range(24)]
    proposal = {'action': 'run', 'arguments': {'operator': 'reference_response',
                'path': 'shared/method.py', 'context': {'estimators': estimators}}}
    result = {'step': 15, 'path': 'shared/method.py', 'proposal': proposal,
              'feedback': {'status': 'inconclusive', 'diagnosis': 'no_untried_supported_setting'}}
    small = {'step': 16, 'path': 'shared/method.py', 'proposal': {'stop': 'no supported setting'},
             'feedback': {'status': 'method_stopped'}}
    request = {'evidence': {'observations_file': 'laboratory_observations.json'},
               'research': {'method_results': [result, small]}}
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    compact = context['research']['method_results'][0]
    reference = compact['proposal']
    assert reference['characters'] == len(json.dumps(proposal))
    assert reference['preview'] == json.dumps(proposal)[:800]
    stored = json.loads((tmp_path / reference['full_content_file']).read_text())
    assert stored == request
    for key in reference['json_path']:
        stored = stored[key]
    assert stored == proposal
    assert compact['feedback'] == result['feedback']
    assert context['research']['method_results'][1]['proposal'] == small['proposal']


def test_rejected_trajectory_action_stays_readable_with_its_error(tmp_path):
    action = {"kind": "trajectory", "sequence": [0.5]}
    feedback = {"error": "each trajectory point must contain only control u"}
    request = {"events": [{"action": "query", "arguments": {"actions": [action]}, "feedback": feedback}]}
    context = prepare_code_context(tmp_path, request, schema={}, instructions="test only")
    assert context["events"][0]["arguments"]["actions"] == [action]
    assert context["events"][0]["feedback"] == feedback


def test_practice_context_keeps_chronology_counterexamples_and_full_trace(tmp_path):
    sequence = [{"u": index / 24} for index in range(24)]
    purchase = {"action": {"kind": "trajectory", "sequence": sequence, "channels": []},
                "row_indices": list(range(72, 96)), "observation_cost": 24, "measurement_cost": 0,
                "rows": [{"features": point, "target": index / 10, "history": sequence[:index]}
                         for index, point in enumerate(sequence)], "measurements": []}
    next_action = {"action": "run", "arguments": {"operator": "refit", "model": "prior.py"}}
    model = {"model": "prior.py", "mse": .2, "evaluated_rows": 72}
    steps = []
    for index in range(12):
        steps.append({"step": index, "action": "query", "arguments": {"path": "method.py"},
                      "feedback": {"program_action": {"action": "finish", "arguments": {"model": "prior.py"}},
                                   "mse": .2, "evaluated_model": model,
                                   "fresh_confirmation": {"passed": False, "mse": .1 + index / 100,
                                                          "rows": 24, "mse_threshold": 1e-10,
                                                          "observation_indices": list(range(72, 96)),
                                                          "observations": [purchase]},
                                   "error": "fresh confirmation failed; new counterexamples are now paid evidence",
                                   "research_advice": {"diagnosis": "response_counterexample",
                                                       "evaluated_model": model, "next_action": next_action,
                                                       "decision": {"selected_action": "refit_response", "reason": "untried_action",
                                                                    "options": [{"observation_cost": 24}], "compute_remaining": 1000}}},
                      "usage": {"backend": "frozen_program_execution", "total_tokens": 0}})
    steps[4]["feedback"]["program_action"] = {"action": "query", "arguments": {"actions": [purchase["action"]]}}
    steps[5]["feedback"].update(returncode=1, stderr="ValueError: unknown operator", status="method_error")
    comparison = {"base": {"train_mse": .084, "heldout_mse": .074},
                  "revision": {"train_mse": .034, "heldout_mse": .064},
                  "selected_refit": {"rows": 96, "rank": 15, "train_mse": .031},
                  "split": {"training_records": [0, 1], "validation_records": [2, 3]},
                  "compute_cost": 2880, "python_execution_seconds": .007}
    steps[6]["feedback"].update(comparison)
    feedback = {"case": "case_2", "directory": "trials/trial_003", "status": "step_budget_exhausted",
                "costs": {"observation_cost": 312, "measurement_cost": 0, "compute_cost": 4992},
                "current_source_coverage": {"confirmed_cases": ["case_0", "case_1"], "missing_cases": ["case_2"]},
                "fresh_confirmation": {"passed": False, "mse": .21, "rows": 24}, "steps": steps}
    request = {"events": [{"action": "run", "arguments": {"operator": "practice", "case": "case_2"}, "feedback": feedback}]}
    original = deepcopy(request)
    trial_path = tmp_path / "trials/trial_003/result.json"
    trial_path.parent.mkdir(parents=True)
    trial_path.write_text(json.dumps(feedback))
    trial_before = trial_path.read_bytes()

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    compact = context["events"][0]["feedback"]
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert (tmp_path / compact["full_trace_file"]).read_bytes() == trial_before
    for key in ("case", "status", "costs", "current_source_coverage", "fresh_confirmation"):
        assert compact[key] == feedback[key]
    assert [step["step"] for step in compact["steps"]] == list(range(12))
    for actual, expected in zip(compact["steps"], steps):
        assert actual["action"] == expected["action"] and actual["arguments"] == expected["arguments"]
        assert actual["feedback"]["program_action"] == expected["feedback"]["program_action"]
        fresh = actual["feedback"]["fresh_confirmation"]
        for key in ("passed", "mse", "rows", "mse_threshold"):
            assert fresh[key] == expected["feedback"]["fresh_confirmation"][key]
        assert actual["feedback"]["error"] == expected["feedback"]["error"]
        assert actual["feedback"]["evaluated_model"] == model
        advice = actual["feedback"]["research_advice"]
        assert advice["next_action"] == next_action
        assert advice["diagnosis"] == "response_counterexample"
        assert advice["decision"] == {"selected_action": "refit_response", "reason": "untried_action"}
        assert "evaluated_model" not in advice
        assert "usage" not in actual and "observation_indices" not in fresh
        record = fresh["observations"][0]
        assert record["row_count"] == record["observation_cost"] == 24
        assert not {"rows_preview", "measurements_preview", "row_indices"} & record.keys()
    error = compact["steps"][5]["feedback"]
    assert error["returncode"] == 1 and error["stderr"] == "ValueError: unknown operator"
    assert error["status"] == "method_error"
    for key, value in comparison.items():
        assert compact["steps"][6]["feedback"][key] == value
    assert len(json.dumps(compact)) < len(json.dumps(feedback)) / 2


def test_practice_advice_keeps_model_outcomes_when_not_duplicated(tmp_path):
    evidence = {"model": "prior.py", "mse": None, "support": "fresh_counterexample", "fresh_mse": .0258}
    feedback = {"steps": [{"step": 0, "feedback": {"research_advice": {"evaluated_model": evidence}}}]}
    request = {"events": [{"arguments": {"operator": "practice"}, "feedback": feedback}]}
    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")
    assert context["events"][0]["feedback"]["steps"][0]["feedback"]["research_advice"]["evaluated_model"] == evidence


def test_world_context_keeps_outcomes_and_physical_citations_without_nested_trace_copies(tmp_path):
    observed = {"index": 17, "lastActionMessage": "crystal activated at measured setting"}
    question = {"id": "fit", "binding_path": "method.py", "status": "refuted",
                "evidence": [17], "answer_observations": [observed]}
    physical = {"id": "physical", "status": "supported", "evidence": [17],
                "answer_observations": [observed]}
    decision = {"questions": [question, physical]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    trace = {"step": 3, "path": "method.py", "proposal": {"actions": [{"action": "TALK", "arg1": 7}]},
             "feedback": {"status": "refuted", "observation_indices": [17],
                          "error": "physical counterexample", "action_results": [observed] * 100,
                          "question": {"reviewed_child_answers": {"old": "previous answer" * 200}}}}
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "files": {"method_decision.json": json.dumps(decision)},
               "research": {"method_results": [trace] * 20, "answers": {"physical": physical}},
               "events": [{"feedback": {"status": "refuted", "steps": [trace] * 20}}]}
    original = deepcopy(request)
    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")
    assert request == original
    saved = json.loads((tmp_path / "research_context.json").read_text())
    assert saved == original
    assert _read_context_reference(tmp_path, context["research"]["answers"]["physical"]) == physical
    assert context["method_decision"]["questions"][1] == physical
    assert len(json.dumps(context)) < len(json.dumps(request)) / 4

    def referenced(value):
        node = json.loads((tmp_path / value["full_content_file"]).read_text())
        for key in value["json_path"]:
            node = node[key]
        return node

    row = context["research"]["method_results"][0]
    assert row["feedback"]["status"] == "refuted"
    assert row["feedback"]["error"] == "physical counterexample"
    assert row["feedback"]["observation_indices"] == [17]
    assert referenced(row) == trace
    assert referenced(context["events"][0]["feedback"]["steps"]) == [trace] * 20
    assert referenced(context["method_decision"]["questions"][0]["answer_observations"]) == [observed]


def test_world_response_answers_compact_cumulative_evidence_without_binding_path(tmp_path):
    observations = [
        {"index": index, "action": {"action": "TALK", "arg1": 42, "arg2": index},
         "feedback": {"success": True},
         "lastActionMessage": "crystal activated" if index % 2 == 0 else "crystal uncalibrated",
         "public_objects": [{"uuid": 42, "frequency": 1000 + index,
                             "activated": index % 2 == 0,
                             "description": "observed reactor control " * 80}]}
        for index in range(20)]
    questions = [
        {"id": f"reference_response_4_{attempt}", "status": "inconclusive",
         "answer": {"diagnosis": "no_untried_supported_setting"},
         "evidence": list(range(20)), "answer_observations": deepcopy(observations)}
        for attempt in range(8)]
    physical = {"id": "physical", "status": "refuted", "evidence": [19],
                "answer_observations": [observations[-1]]}
    decision = {"questions": [*questions, physical]}
    decision_file = tmp_path / "method_decision.json"
    decision_file.write_text(json.dumps(decision))
    before = decision_file.read_bytes()
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert decision_file.read_bytes() == before
    compact_physical = context["method_decision"]["questions"][-1]
    assert {key: value for key, value in compact_physical.items() if key != "answer_observations"} == {
        key: value for key, value in physical.items() if key != "answer_observations"}
    _assert_world_snapshot_reference(tmp_path, compact_physical["answer_observations"][0], observations[-1])
    for index, question in enumerate(context["method_decision"]["questions"][:-1]):
        assert "binding_path" not in question
        assert question["status"] == questions[index]["status"]
        assert question["answer"] == questions[index]["answer"]
        assert question["evidence"] == list(range(20))
        evidence = question["answer_observations"]
        assert evidence["observation_indices"] == list(range(20))
        assert len(evidence["observations_preview"]) == (4 if index == 0 else 0)
        if index == 0:
            for preview, row_index in zip(evidence["observations_preview"], (0, 1, 18, 19)):
                _assert_world_snapshot_reference(tmp_path, preview, observations[row_index])
            assert {row["lastActionMessage"] for row in evidence["observations_preview"]} == {
                "crystal activated", "crystal uncalibrated"}
        stored = json.loads((tmp_path / evidence["full_content_file"]).read_text())
        for key in evidence["json_path"]:
            stored = stored[key]
        assert stored == observations
    assert len(json.dumps(context)) < len(json.dumps(original)) / 3


def _read_context_reference(workspace, descriptor):
    stored = json.loads((workspace / descriptor["full_content_file"]).read_text())
    for key in descriptor["json_path"]:
        stored = stored[key]
    return stored


def test_large_response_history_members_are_retrievable_without_hiding_research_decisions(tmp_path):
    hypotheses = [{"id": f"member_{index}", "lower": -.5, "upper": .5,
        "program": {"op": "add", "children": [
            {"op": "const", "value": index + term / 100.} for term in range(32)]},
        "predictions_by_crystal": {"1": 10., "2": 20., "3": 30.}}
        for index in range(24)]
    assert len(json.dumps(hypotheses)) > 12000
    trial = {"number": 3, "prediction": 31., "procedure": {"path": "shared/research.py"}}
    obligation = {"reason": "stagnation", "conditional_operational_search": {
        "remaining_settings": 19, "suggested_trial": {"selected_experiment": trial}}}
    row = {"hypotheses": hypotheses, "selected_experiment": trial,
        "selection": {"selected_experiment": trial}, "evidence": [17, 18],
        "response_settings": [{"crystal": 3, "frequency": 30., "activated": False,
                               "evidence": [18]}],
        "family_research": {"families": [{"id": "cubic", "status": "SAT_WITNESS"}],
                            "admitted_signatures": [[False]], "research_obligation": obligation},
        "conditional_simplification": {"adopted": False, "status": "family_research_needed",
            "assumption": {"supports": [{"id": "cubic_term"}]},
            "family_research": {"research_obligation": obligation}},
        "research_obligation": obligation}
    decision = {"questions": [{"id": "response", "status": "inconclusive",
        "answer": {"response_research_history": [deepcopy(row)]}}]}
    decision_file = tmp_path / "method_decision.json"
    decision_file.write_text(json.dumps(decision))
    decision_bytes = decision_file.read_bytes()
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "research": {"response_research_history": [deepcopy(row)]},
        "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    histories = [(context["research"]["response_research_history"], "research_context.json"),
        (context["method_decision"]["questions"][0]["answer"]["response_research_history"],
         "method_decision.json")]
    for history, filename in histories:
        compact = history[0]
        descriptor = compact["hypotheses"]
        assert descriptor["count"] == len(hypotheses)
        assert descriptor["full_content_file"] == filename
        assert _read_context_reference(tmp_path, descriptor) == hypotheses
        assert {key: value for key, value in compact.items() if key != "hypotheses"} == {
            key: value for key, value in row.items() if key != "hypotheses"}
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert decision_file.read_bytes() == decision_bytes


def test_small_history_and_large_current_or_authored_hypotheses_stay_inline(tmp_path):
    small = [{"id": "linear", "program": {"op": "feature", "name": "x"}}]
    authored = [{"id": "new_abstraction", "claim": "A testable research hypothesis. " * 500}]
    assert len(json.dumps(authored)) > 12000
    research = {"response_research_history": [{"hypotheses": small, "evidence": [1]}],
                "response_research": {"hypotheses": authored},
                "authored_analysis": {"hypotheses": authored}}
    decision = {"questions": [{"id": "authored", "answer": {
        "hypotheses": authored, "response_research_history": [{"hypotheses": small}]}}]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "research": research, "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    assert context["research"] == research
    assert context["method_decision"] == decision
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original


def _assert_world_snapshot_reference(workspace, preview, original):
    assert not {"public_objects", "public_relations", "dialog_box"} & preview.keys()
    assert {key: value for key, value in preview.items()
            if key not in ("full_content_file", "json_path")} == {
        key: value for key, value in original.items()
        if key not in ("public_objects", "public_relations", "dialog_box")}
    assert _read_context_reference(workspace, preview) == original


def test_nested_world_experiment_snapshots_and_mirrored_focus_keep_actual_outcomes(tmp_path):
    rows = [{'index': index, 'lastActionMessage': 'The reactor is uncalibrated'}
            for index in range(95)]
    focus = {'kind': 'parent_research', 'stage': 'contrast',
             'answer_observations': rows, 'changed_child_ids': ['scan']}
    last_action = {'index': 94, 'action': {'chosen_dialog_option_int': 3},
                   'feedback': {'success': True, 'errors': []},
                   'observation': {'public_objects': ['current physical state'] * 1000}}
    request = {'evidence': {'observations_file': 'laboratory_observations.json'},
               'research': {'research_focus': deepcopy(focus)}, 'research_focus': focus,
               'current_observation': deepcopy(last_action['observation']),
               'events': [{'feedback': {'experiment': {'last_action': deepcopy(last_action),
                           'after_observation_error': 'analysis needs repair'}}} for _ in range(8)]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    assert _read_context_reference(tmp_path, context['research']['research_focus']) == focus
    preview = context['research_focus']['answer_observations']
    assert preview['observations_preview'] == [rows[i] for i in (0, 1, 93, 94)]
    assert _read_context_reference(tmp_path, preview) == rows
    for event in context['events']:
        experiment = event['feedback']['experiment']
        assert experiment['after_observation_error'] == 'analysis needs repair'
        compact = experiment['last_action']
        assert {key: value for key, value in compact.items() if key != 'observation'} == {
            key: value for key, value in last_action.items() if key != 'observation'}
        assert _read_context_reference(tmp_path, compact['observation']) == last_action['observation']
    assert context['current_observation'] == original['current_observation']
    assert request == original
    assert json.loads((tmp_path / 'research_context.json').read_text()) == original
    assert len(json.dumps(context)) < len(json.dumps(original)) / 3


def test_long_world_method_trace_keeps_recent_outcomes_and_complete_retrievable_history(tmp_path):
    results = [{'step': index, 'path': 'shared/scan.py',
                'proposal': {'actions': [{'chosen_dialog_option_int': 3}]},
                'feedback': {'observation_indices': [index], 'status': 'executed'}}
               for index in range(130)]
    results[0]['feedback'] = {'status': 'method_stopped', 'error': 'initial control failed'}
    request = {'evidence': {'observations_file': 'laboratory_observations.json'},
               'research': {'method_results': results},
               'research_focus': {'previous_execution': deepcopy(results[0])}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    history = context['research']['method_results_history']
    assert history['count'] == 130 and history['preview_start'] == 118
    assert _read_context_reference(tmp_path, history) == results
    assert [row['step'] for row in context['research']['method_results']] == list(range(118, 130))
    for compact, original_row in zip(context['research']['method_results'], results[-12:]):
        assert compact['feedback'] == original_row['feedback']
        assert _read_context_reference(tmp_path, compact) == original_row
    assert context['research_focus']['previous_execution'] == results[0]
    assert request == original
    assert json.loads((tmp_path / 'research_context.json').read_text()) == original


def test_short_large_world_evidence_compacts_local_research_views_but_keeps_small_answers(tmp_path):
    rows = [{"index": index, "action": {"action": "TALK", "arg1": 42},
             "feedback": {"success": True}, "lastActionMessage": "The reactor is uncalibrated",
             "public_objects": [{"uuid": 42, "description": "observed reactor control " * 80,
                                 "frequency": 1000 + index}]}
            for index in range(9)]
    assert len(json.dumps(rows)) > 12000
    question = {"id": "reference", "status": "inconclusive", "evidence": list(range(9)),
                "answer_observations": rows}
    small = {"id": "reference_state", "status": "refuted", "evidence": [8],
             "answer_observations": [{"index": 8, "lastActionMessage": "The reactor is uncalibrated"}]}
    decision = {"questions": [deepcopy(question), deepcopy(small)]}
    decision_file = tmp_path / "method_decision.json"
    decision_file.write_text(json.dumps(decision))
    before = decision_file.read_bytes()
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "research_focus": {"question": deepcopy(question), "returned_children": [deepcopy(small)]},
               "llm_inquiry": {"question": deepcopy(question)},
               "procedure_repair": {"latest_execution": {"question": deepcopy(question)}},
               "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    assert _read_context_reference(tmp_path, context["research_focus"]["question"]) == question
    references = [context["llm_inquiry"]["question"]["answer_observations"],
                  context["procedure_repair"]["latest_execution"]["question"]["answer_observations"],
                  context["method_decision"]["questions"][0]["answer_observations"]]
    for index, reference in enumerate(references):
        assert reference["observation_indices"] == list(range(9))
        assert len(reference["observations_preview"]) == (4 if index == 0 else 0)
        if index == 0:
            for preview, row_index in zip(reference["observations_preview"], (0, 1, 7, 8)):
                _assert_world_snapshot_reference(tmp_path, preview, rows[row_index])
        assert _read_context_reference(tmp_path, reference) == rows
    assert _read_context_reference(tmp_path, context["research_focus"]["returned_children"][0]) == small
    assert context["method_decision"]["questions"][1] == small
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert decision_file.read_bytes() == before


def test_physical_preview_deduplicates_equal_records_without_merging_changed_content_at_same_index(tmp_path):
    rows = [{"index": index, "lastActionMessage": "Observed state", "feedback": {"success": True}}
            for index in range(13)]
    changed = deepcopy(rows)
    changed[0]["lastActionMessage"] = "Different recorded state at the same index"
    original_question = {"id": "old", "status": "inconclusive", "answer_observations": rows}
    revised_question = {"id": "current", "status": "refuted", "answer_observations": changed}
    decision = {"questions": [deepcopy(revised_question)]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "research": {"answers": {"old": deepcopy(original_question)}},
               "events": [{"feedback": {"question": deepcopy(original_question)}}],
               "research_focus": {"question": deepcopy(revised_question)},
               "files": {"method_decision.json": json.dumps(decision)}}

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    original = context["research"]["answers"]["old"]["answer_observations"]
    repeated = context["events"][0]["feedback"]["question"]["answer_observations"]
    assert _read_context_reference(tmp_path, context["research_focus"]["question"]) == revised_question
    revision = context["method_decision"]["questions"][0]["answer_observations"]
    assert original["observations_preview"] == [rows[i] for i in (0, 1, 11, 12)]
    assert repeated["observations_preview"] == []
    assert revision["observations_preview"] == [changed[0]]
    for reference, expected in ((original, rows), (repeated, rows),
                                (revision, changed)):
        assert reference["observation_indices"] == list(range(13))
        assert _read_context_reference(tmp_path, reference) == expected


def test_world_question_mirror_references_exact_decision_entry_and_retains_different_state(tmp_path):
    rows = [{"index": index, "lastActionMessage": "active" if index % 2 else "inactive"}
            for index in range(13)]
    shared = {"id": "shared", "status": "supported", "question": "Does the reference respond?",
              "answer": {"active": True}, "evidence": list(range(13)), "answer_observations": rows}
    different = {"id": "changed", "status": "refuted", "answer": {"active": False}, "evidence": [3]}
    absent = {"id": "unlisted", "status": "open", "question": "Which observation is needed?"}
    trials = [{'question_id': 'shared', 'hypotheses': [{'id': 'active'}, {'id': 'inactive'}],
               'selected_experiment': {'id': 'probe'}, 'observation_indices': list(range(13))}]
    detailed = {**deepcopy(shared), "procedure_action": {"action": "run", "arguments": {
        "operator": "procedure", "path": "shared/reference.py"}}, 'experiment_trials': deepcopy(trials)}
    # The matching entry is deliberately second, so lookup must use the ID
    # and its actual file position rather than the state's list position.
    decision = {"questions": [{**deepcopy(different), "answer": {"active": True}}, detailed]}
    decision_file = tmp_path / "method_decision.json"
    decision_file.write_text(json.dumps(decision))
    before = decision_file.read_bytes()
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "research": {"questions": [shared, different, absent], 'experiment_trials': trials,
                            'answers': {q['id']: {key: q[key] for key in ('status', 'answer', 'evidence')
                                                if key in q} for q in (shared, different, absent)}},
               'research_focus': {'prior_trials': deepcopy(trials)},
               "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    reference, kept_different, kept_absent = context["research"]["questions"]
    assert reference == {"id": "shared", "status": "supported", "full_content_file": "method_decision.json",
                         "json_path": ["questions", 1]}
    assert _read_context_reference(tmp_path, reference) == detailed
    assert kept_different == different and kept_absent == absent
    assert context["method_decision"]["questions"][0] == decision["questions"][0]
    surviving = context["method_decision"]["questions"][1]
    assert {key: value for key, value in surviving.items() if key not in ("answer_observations", 'experiment_trials')} == {
        key: value for key, value in detailed.items() if key not in ("answer_observations", 'experiment_trials')}
    answers = context['research']['answers']
    assert _read_context_reference(tmp_path, answers['shared']) == detailed
    assert answers['changed'] == original['research']['answers']['changed']
    assert answers['unlisted'] == original['research']['answers']['unlisted']
    assert context['research']['experiment_trials'] == trials
    assert _read_context_reference(tmp_path, surviving['experiment_trials']) == trials
    assert _read_context_reference(tmp_path, context['research_focus']['prior_trials']) == trials
    # Removing the state mirror must not consume the only physical preview.
    evidence = surviving["answer_observations"]
    assert evidence["observations_preview"] == [rows[index] for index in (0, 1, 11, 12)]
    assert evidence["observation_indices"] == list(range(13))
    assert _read_context_reference(tmp_path, evidence) == rows
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert decision_file.read_bytes() == before


def test_returned_child_mirrors_reuse_decision_answers_without_hiding_changed_returns(tmp_path):
    child = {"id": "response", "status": "refuted", "evidence": [7, 9],
             "answer": {"activated": False, "predicted_frequency": 1686},
             "experiment_choice": {"predictions": {"local": True, "alternative": False}}}
    changed = {"id": "changed", "status": "inconclusive", "answer": {"activated": None}}
    absent = {"id": "new", "status": "open", "question": "What additional reading is needed?"}
    detailed = {**deepcopy(child), "analysis_path": "shared/response.py"}
    decision = {"questions": [{**deepcopy(changed), "answer": {"activated": True}}, detailed]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "research_focus": {"returned_children": [child, changed, absent],
                                  "changed_child_ids": ["response"]},
               "current_observation": {"dialog_box": {"options": {"1": "Increase frequency"}}},
               "events": [{"action": "evaluate", "arguments": {"question_id": "response"},
                           "feedback": deepcopy(child)}],
               "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    reference, kept_changed, kept_absent = context["research_focus"]["returned_children"]
    assert reference == {"id": "response", "status": "refuted", "full_content_file": "method_decision.json",
                         "json_path": ["questions", 1]}
    assert _read_context_reference(tmp_path, reference) == detailed
    assert kept_changed == changed and kept_absent == absent
    assert context["method_decision"] == decision
    assert context["research_focus"]["changed_child_ids"] == ["response"]
    assert context["current_observation"] == original["current_observation"]
    assert context["events"] == original["events"]
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original


def test_response_research_mirrors_reference_exact_canonical_answer_fields(tmp_path):
    answer = {"family_research": {
        "families": [{"id": "cubic", "status": "SAT_WITNESS"}],
        "possible_signatures": [[True], [False]], "evidence": [3, 7]},
        "conditional_simplification": {"adopted": False, "assumption": {"basis": ["x3"]},
            "family_research": {"research_obligation": {"remaining_settings": 31}}}}
    row = {"question_id": "response", "status": "family_research_needed", **deepcopy(answer),
           "selected_experiment": {"number": 5, "prediction": 6171.},
           "hypotheses": [{"id": "author_explanation", "claim": "Preserve this separate claim"}],
           "evidence": [3, 7]}
    decision = {"questions": [{"id": "other", "answer": {}},
        {"id": "response", "status": "inconclusive", "answer": deepcopy(answer)}]}
    decision_file = tmp_path / "method_decision.json"
    decision_file.write_text(json.dumps(decision))
    original_bytes = decision_file.read_bytes()
    view = {"response_research_history": [deepcopy(row)], "response_research": deepcopy(row)}
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "events": [{"feedback": deepcopy(view)}],
        **{name: deepcopy(view) for name in ("research", "research_focus", "procedure_repair", "llm_inquiry")},
        "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    for compact_view in (context["events"][0]["feedback"], *(context[name] for name in
                         ("research", "research_focus", "procedure_repair", "llm_inquiry"))):
        for compact in (*compact_view["response_research_history"], compact_view["response_research"]):
            for field in answer:
                assert compact[field] == {"full_content_file": "method_decision.json",
                    "json_path": ["questions", 1, "answer", field]}
                assert _read_context_reference(tmp_path, compact[field]) == answer[field]
            assert {key: value for key, value in compact.items() if key not in answer} == {
                key: value for key, value in row.items() if key not in answer}
    assert context["method_decision"] == decision
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert decision_file.read_bytes() == original_bytes


def test_response_research_mirrors_preserve_changed_fields_and_unmatched_questions(tmp_path):
    answer = {"family_research": {"families": [{"id": "cubic", "status": "SAT_WITNESS"}]},
              "conditional_simplification": {"adopted": False, "evidence": [7]}}
    changed_family = {"question_id": "response", **deepcopy(answer)}
    changed_family["family_research"]["families"][0]["status"] = "UNSAT_VERIFIED"
    changed_condition = {"question_id": "response", **deepcopy(answer)}
    changed_condition["conditional_simplification"]["evidence"].append(8)
    absent = {"question_id": "new_question", **deepcopy(answer)}
    unbound = deepcopy(answer)
    decision = {"questions": [{"id": "response", "answer": deepcopy(answer)}]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "research": {"response_research_history": [changed_family, absent, unbound],
                     "response_research": changed_condition,
                     "hypotheses": [{"id": "unrelated_author_hypothesis", "claim": "Retain me"}]},
        "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    history = context["research"]["response_research_history"]
    assert history[0]["family_research"] == changed_family["family_research"]
    assert _read_context_reference(tmp_path, history[0]["conditional_simplification"]) == answer["conditional_simplification"]
    assert history[1:] == [absent, unbound]
    current = context["research"]["response_research"]
    assert current["conditional_simplification"] == changed_condition["conditional_simplification"]
    assert _read_context_reference(tmp_path, current["family_research"]) == answer["family_research"]
    assert context["research"]["hypotheses"] == request["research"]["hypotheses"]
    assert context["method_decision"] == decision
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original


def test_past_response_answers_keep_decisions_and_raw_differences_while_latest_stays_complete(tmp_path):
    family = {"id": "cubic", "status": "SAT_WITNESS",
              "basis": [{"op": "feature", "name": "signal"}], "source_ids": ["authored"]}
    research = {"scope": "shared_response_family", "response_count": 31,
        "nonlocal_progress": 0, "no_progress_cycles": 2, "selected_role": "shared_premise_research",
        "selected_test": None, "research_obligation": {"question": "Revise the shared band?"},
        "families": [family], "admitted_signatures": [[True, False]],
        "old_only_diagnostic": {"counterexample": {"crystal": 4, "frequency": 1012.}}}
    conditional = {"status": "family_research_needed", "adopted": False,
        "assumption": {"supports": ["cubic_only"]},
        "family_research": {"old_only_bound": [-1., 3.]}}
    old = {"id": "response_old", "question": "Which response mechanism survives?",
        "status": "inconclusive", "evidence": [2, 7],
        "answer": {"family_research": research, "conditional_simplification": conditional,
                   "activated": False, "selected_setting": 1012.}}
    latest = {"id": "response_latest", "status": "inconclusive", "evidence": [2, 7, 9],
        "answer": {"family_research": {**deepcopy(research), "response_count": 32,
                                       "admitted_signatures": [[False, False]]},
                   "conditional_simplification": deepcopy(conditional)}}
    ordinary = {"id": "instrument", "status": "supported", "evidence": [9],
                "answer": {"instrument_name": "spectrometer"}}
    decision = {"questions": [old, latest, ordinary]}
    decision_file = tmp_path / "method_decision.json"
    decision_file.write_text(json.dumps(decision))
    original_bytes = decision_file.read_bytes()
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    shown_old, shown_latest, shown_ordinary = context["method_decision"]["questions"]
    assert shown_latest == latest
    assert shown_ordinary == ordinary
    assert {key: value for key, value in shown_old.items() if key != "answer"} == {
        key: value for key, value in old.items() if key != "answer"}
    assert shown_old["answer"]["activated"] is False
    assert shown_old["answer"]["selected_setting"] == 1012.
    summary = shown_old["answer"]["family_research"]
    assert _read_context_reference(tmp_path, summary) == research
    assert "old_only_diagnostic" not in summary
    for key in ("scope", "response_count", "nonlocal_progress", "no_progress_cycles",
                "selected_role", "selected_test", "research_obligation"):
        assert summary[key] == research[key]
    assert _read_context_reference(tmp_path, summary["families"][0]["basis"]) == family["basis"]
    assert {key: value for key, value in summary["families"][0].items() if key != "basis"} == {
        key: value for key, value in family.items() if key != "basis"}
    child = shown_old["answer"]["conditional_simplification"]
    assert _read_context_reference(tmp_path, child) == conditional
    for key in ("status", "adopted", "assumption"):
        assert child[key] == conditional[key]
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert decision_file.read_bytes() == original_bytes


def test_long_response_research_keeps_predictions_and_retrievable_solver_and_world_evidence(tmp_path):
    # The failed live continuation carried 18 rounds of 48 solver diagnostics,
    # alongside cumulative physical answers. No generated code is needed to
    # reproduce the oversized evidence payload.
    observations = [{
        "index": index, "action": {"chosen_dialog_option_int": 3},
        "feedback": {"success": True, "errors": []},
        "lastActionMessage": f"Crystal Reactor #5 is uncalibrated at {1000 + index} Hertz.",
        "public_objects": [{"uuid": object_id, "name": f"crystal {object_id}",
                            "description": "A measured crystal in the laboratory."}
                           for object_id in range(8)],
        "public_relations": [{"subject": 5, "relation": "in", "object": 42,
                              "text": "in crystal reactor (uncalibrated)"}],
        "dialog_box": {"text": f"Current frequency: {1000 + index} Hertz",
                       "options": {"1": "Increase by 1000 Hz", "2": "Increase by 100 Hz",
                                   "3": "Increase by 1 Hz", "4": "Exit"}},
    } for index in range(504)]
    questions = []
    for attempt in range(18):
        diagnostics = [{
            "family": f"spectral_multichannel_linear_{family}",
            "status": "incomplete" if family % 2 else "feasible",
            "branch_count": 19000 + attempt * 100 + family,
            "branches_attempted": 128, "branches_feasible": 53,
            "lp_calls": 234, "lp_iterations": 2112 + family,
            "solver_statuses": {"0": 190, "2": 44}, "verification_failures": 0,
            "branch_limit": 128, "limited_by_branch_budget": bool(family % 2),
            "scope": "reachable_untried_target_support_in_declared_basis",
            "models_retained": family % 3, "design_target_number": 4 + family % 2,
        } for family in range(48)]
        rows = observations[:300 + attempt * 12]
        questions.append({
            "id": f"reference_response_{attempt}", "status": "refuted",
            "answer": {"diagnosis": "physical_prediction_refuted",
                       "selected_setting": 1686 + attempt, "selected_number": 5,
                       "activated": False, "joint_parameter_search": diagnostics},
            "evidence": [row["index"] for row in rows], "answer_observations": rows,
        })
    decision = {"questions": questions}
    decision_file = tmp_path / "method_decision.json"
    decision_file.write_text(json.dumps(decision))
    before = decision_file.read_bytes()
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "current_observation": observations[-1],
               "files": {"method_decision.json": json.dumps(decision)}}
    original = deepcopy(request)
    assert len(json.dumps(request)) > 1048576

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    assert len(json.dumps(context)) < 1048576
    assert context["current_observation"] == observations[-1]
    for expected, compact in zip(questions, context["method_decision"]["questions"]):
        assert compact["status"] == "refuted"
        assert compact["evidence"] == expected["evidence"]
        assert {key: value for key, value in compact["answer"].items()
                if key != "joint_parameter_search"} == {
            key: value for key, value in expected["answer"].items()
            if key != "joint_parameter_search"}
        diagnostics = compact["answer"]["joint_parameter_search"]
        assert diagnostics["count"] == 48
        assert diagnostics["status_counts"] == {"feasible": 24, "incomplete": 24}
        assert diagnostics["limited_by_branch_budget_count"] == 24
        assert diagnostics["models_retained"] == 48
        assert _read_context_reference(tmp_path, diagnostics) == expected["answer"]["joint_parameter_search"]
        evidence = compact["answer_observations"]
        assert evidence["observation_indices"] == expected["evidence"]
        assert _read_context_reference(tmp_path, evidence) == expected["answer_observations"]
        for preview in evidence["observations_preview"]:
            _assert_world_snapshot_reference(tmp_path, preview, observations[preview["index"]])
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original
    assert decision_file.read_bytes() == before


def test_short_world_answer_references_snapshots_but_keeps_small_diagnostics_and_plain_rows(tmp_path):
    physical = {"index": 62, "action": {"action": "TALK", "arg1": 42},
                "feedback": {"success": True}, "lastActionMessage": "No activation at 1686 Hz",
                "measured_frequency": 1686, "public_objects": [{"uuid": 42, "name": "reactor"}],
                "public_relations": [], "dialog_box": {"options": {"1": "Exit"}}}
    plain = {"index": 63, "lastActionMessage": "Still uncalibrated", "feedback": {"success": True}}
    diagnostics = [{"family": "linear", "status": "incomplete", "models_retained": 1}]
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "research_focus": {"answer_observations": [physical, plain],
                                  "answer": {"joint_parameter_search": diagnostics}},
               "current_observation": physical}

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    compact, kept = context["research_focus"]["answer_observations"]
    _assert_world_snapshot_reference(tmp_path, compact, physical)
    assert kept == plain
    assert context["research_focus"]["answer"]["joint_parameter_search"] == diagnostics
    assert context["current_observation"] == physical


def test_computed_response_pool_keeps_predictions_conflicts_and_executable_source(tmp_path):
    candidates = [{"id": str(i), "family": "cubic", "status": "rejected",
        "target_prediction": 666., "reason": "paid_negative", "conflicting_responses": [
            {"crystal": 5, "frequency": 666., "activated": False, "evidence": [350]}],
        "program": {"op": "add", "children": [{"op": "const", "value": j} for j in range(100)]}}
        for i in range(20)]
    comparison = {"candidates": candidates, "family_research": {"families": [
        {"id": "cubic", "status": "SAT_WITNESS"}]}, "new_responses": [
        {"crystal": 5, "frequency": 666., "activated": False, "evidence": [350]}],
        "experiment_action": {"action": "run", "arguments": {"experiment": {"setting": 777}}}}
    decision = {"explanation_proposals": [{"comparison_result": deepcopy(comparison)}]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "files": {"method_decision.json": json.dumps(decision)},
        "events": [{"action": "response_return", "feedback": {"response_comparison": deepcopy(comparison)}}]}
    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")
    copies = [context["events"][0]["feedback"]["response_comparison"],
              context["method_decision"]["explanation_proposals"][0]["comparison_result"]]
    for compact in copies[:1]:
        pool = compact["candidates"]
        assert pool["count"] == len(pool["preview"]) == len(candidates)
        assert _read_context_reference(tmp_path, pool) == candidates
        for original, preview in zip(candidates, pool["preview"]):
            assert preview == {k: v for k, v in original.items() if k != "program"}
        assert {k: v for k, v in compact.items() if k != "candidates"} == {
            k: v for k, v in comparison.items() if k != "candidates"}
    repeated = copies[1]
    assert 'candidates' not in repeated
    assert _read_context_reference(tmp_path, repeated) == comparison
    for key in ('family_research', 'new_responses', 'experiment_action'):
        assert repeated[key] == comparison[key]
    assert json.loads((tmp_path / "research_context.json").read_text()) == request


def test_author_comparison_history_retains_counterexamples_and_latest_search(tmp_path):
    old = {"binding_path": "shared/proposal.py", "status": "experiment_selected",
        "comparison_status": "inconclusive", "response_count": 32,
        "family_research": {"families": [{"id": "cubic", "status": "SAT_WITNESS"}]},
        "new_responses": [{"frequency": 666., "activated": False, "evidence": [350]}],
        "experiment_action": {"action": "run", "arguments": {"experiment": {"setting": 777.}}},
        "selection": {"details": "solver output " * 1500}}
    current = {**deepcopy(old), "comparison_status": "refuted", "response_count": 33,
        "family_research": {"families": [{"id": "cubic", "status": "UNSAT_VERIFIED"}]},
        "new_responses": [{"frequency": 777., "activated": False, "evidence": [360]}]}
    physical = {"prediction_status": "refuted", "question_id": "point_666"}
    request = {"evidence": {"observations_file": "laboratory_observations.json"}, "events": [
        {"action": "response_return", "feedback": {"response_comparison": old, "physical_test": physical}},
        {"action": "response_return", "feedback": {"response_comparison": current}}]}
    original = deepcopy(request)
    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")
    older, latest = context['events']
    summary = older['feedback']['response_comparison']
    assert _read_context_reference(tmp_path, summary) == old
    assert {k: v for k, v in summary.items() if k not in ('full_content_file', 'json_path')} == {
        k: v for k, v in old.items() if k != 'selection'}
    assert older['feedback']['physical_test'] == physical
    assert latest['feedback']['response_comparison'] == current
    assert request == original
    assert json.loads((tmp_path / 'research_context.json').read_text()) == original


@pytest.mark.parametrize("latest_kind", ["direct", "nested"])
@pytest.mark.parametrize("binding_path", ["shared/proposal.py", None])
def test_mixed_response_history_keeps_latest_comparison_and_retrievable_past_verdicts(
        tmp_path, latest_kind, binding_path):
    candidates = [{"id": f"member_{i}", "family": "authored", "status": "rejected",
        "target_prediction": 700. + i, "fit_mse": .2,
        "reason": "paid_negative", "conflicting_responses": [
            {"crystal": 5, "frequency": 700. + i, "activated": False, "evidence": [350 + i]}],
        "program": {"op": "add", "children": [
            {"op": "const", "value": j} for j in range(100)]}}
        for i in range(20)]
    assert len(json.dumps(candidates)) > 12000

    def comparison(index):
        return {"operator": "reference_response", **({"binding_path": binding_path} if binding_path else {}),
            "status": "family_research_needed", "comparison_status": "inconclusive",
            "question_id": "same_representation", "response_count": 247 + index,
            "family_research": {"families": [{"id": "authored", "status": "SAT_WITNESS"}],
                "no_progress_cycles": 2 + index, "selected_role": "shared_premise_research",
                "research_obligation": {"reason": "no_nonlocal_progress"}},
            "new_responses": [{"crystal": 5, "frequency": 727. + index,
                               "activated": False, "evidence": [390 + index]}],
            "evidence": [350, 390 + index], "experiment_action": None,
            "revision_action": {"action": "run", "arguments": {
                "operator": "propose_explanations", "path": "shared/proposal.py"}},
            "premise_design_action": {"action": "run", "arguments": {
                "operator": "propose_explanations", "purpose": "design premise test"}},
            "candidates": deepcopy(candidates),
            "selection": {"detail": f"comparison {index} retained search"}}

    comparisons = [comparison(i) for i in range(3)]
    physical = {"prediction_status": "refuted", "question_id": "actual_point",
                "observed_responses": comparisons[1]["new_responses"]}
    unrelated = {"operator": "ordinary_diagnostic", "binding_path": "shared/other.py",
                 "status": "calculated", "candidates": deepcopy(candidates)}
    events = [
        {"action": "run", "feedback": deepcopy(comparisons[0])},
        {"action": "response_return", "feedback": {"operator": "reference_response",
            "response_comparison": deepcopy(comparisons[1]), "physical_test": physical}},
        {"action": "run", "feedback": unrelated},
        {"action": "run" if latest_kind == "direct" else "response_return",
         "feedback": (deepcopy(comparisons[2]) if latest_kind == "direct" else
                      {"response_comparison": deepcopy(comparisons[2])})}]
    request = {"evidence": {"observations_file": "laboratory_observations.json"}, "events": events}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    older = [context["events"][0]["feedback"],
             context["events"][1]["feedback"]["response_comparison"]]
    for summary, full in zip(older, comparisons):
        assert _read_context_reference(tmp_path, summary) == full
        for key in ("status", "comparison_status", "binding_path", "question_id", "response_count",
                    "family_research", "new_responses", "evidence", "experiment_action", "revision_action",
                    "premise_design_action"):
            assert summary.get(key) == full.get(key)
        assert "candidates" not in summary and "selection" not in summary
    assert context["events"][1]["feedback"]["physical_test"] == physical
    assert context["events"][2]["feedback"] == unrelated
    latest = context["events"][3]["feedback"]
    if latest_kind == "nested":
        latest = latest["response_comparison"]
    pool = latest["candidates"]
    assert pool["count"] == len(pool["preview"]) == len(candidates)
    assert _read_context_reference(tmp_path, pool) == candidates
    for preview, full in zip(pool["preview"], candidates):
        assert preview == {key: value for key, value in full.items() if key != "program"}
    assert {key: value for key, value in latest.items() if key != "candidates"} == {
        key: value for key, value in comparisons[2].items() if key != "candidates"}
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original


@pytest.mark.parametrize("latest_kind", ["direct", "proposal"])
def test_proposal_returns_share_chronological_response_history(tmp_path, latest_kind):
    comparisons = [{"operator": "reference_response", "status": "family_research_needed",
        "comparison_status": "inconclusive", "response_count": 350 + index,
        "family_research": {"families": [{"id": "cubic", "status": "SAT_WITNESS"}]},
        "new_responses": [{"frequency": 589. + index, "activated": False,
                           "evidence": [568 + index]}],
        "evidence": [568 + index],
        "experiment_action": {"action": "run", "arguments": {"experiment": {"setting": 590. + index}}},
        "revision_action": {"action": "run", "arguments": {"operator": "propose_explanations"}},
        "experiments": [{"id": f"next_{index}", "predictions": {"a": True, "b": False}}],
        "selection": {"details": f"retained comparison {index} " * 1000}}
        for index in range(3)]

    def event(comparison, kind):
        return {"action": "run", "feedback": deepcopy(comparison) if kind == "direct" else {
            "operator": "propose_explanations", "model_calls": 1,
            "proposal": {"comparison_result": deepcopy(comparison), "condition_research": {},
                         "draft_changed": False, "author_work_attempted": False}}}

    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "events": [event(comparisons[0], "proposal"), event(comparisons[1], "direct"),
                          event(comparisons[2], latest_kind)]}
    original = deepcopy(request)
    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")
    shown = [item["feedback"].get("proposal", {}).get("comparison_result", item["feedback"])
             for item in context["events"]]
    for summary, full in zip(shown[:-1], comparisons[:-1]):
        assert _read_context_reference(tmp_path, summary) == full
        for key in ("status", "comparison_status", "response_count", "family_research",
                    "new_responses", "evidence", "experiment_action", "revision_action"):
            assert summary[key] == full[key]
        assert "selection" not in summary
    assert shown[0]["json_path"] == ["events", 0, "feedback", "proposal", "comparison_result"]
    assert shown[-1] == comparisons[-1]
    assert context["events"][0]["feedback"]["proposal"]["draft_changed"] is False
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original


def test_ordinary_response_search_references_duplicate_pools_and_answers_without_losing_decisions(tmp_path):
    program = {"op": "add", "children": [{"op": "const", "value": i} for i in range(120)]}
    candidates = [{"id": f"member_{i}", "family": "authored", "status": "rejected",
        "target_prediction": 700. + i, "reason": "paid_negative", "conflicting_responses": [
            {"frequency": 700. + i, "activated": False, "evidence": [390 + i]}],
        "program": deepcopy(program)} for i in range(20)]
    hypotheses = [{"id": f"response_{i}", "candidate_id": f"member_{i}",
        "lower": -1., "upper": 1., "target_prediction": 700. + i,
        "predictions_by_crystal": {"1": 20., "5": 700. + i}, "program": deepcopy(program)}
        for i in range(20)]
    selected = {"number": 5, "prediction": 711.,
                "predictions": {row["id"]: False for row in hypotheses}}
    answer = {"family_research": {"selected_role": "shared_premise_research",
                                  "no_progress_cycles": 3},
              "conditional_simplification": {"status": "not_adopted"}}
    feedback = {"operator": "reference_response", "question_id": "ordinary_response",
        "status": "family_research_needed", **deepcopy(answer),
        "candidates": candidates, "hypotheses": hypotheses,
        "operational_fallback": {"diagnosis": "finite_search", "selected_experiment": selected,
            "predicted_separation": .25, "hypotheses": deepcopy(hypotheses)},
        "premise_design_action": {"action": "run", "arguments": {
            "operator": "propose_explanations", "purpose": "design premise test"}}}
    decision = {"questions": [{"id": "ordinary_response", "status": "answered", "answer": answer}]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "files": {"method_decision.json": json.dumps(decision)},
        "events": [{"action": "run", "feedback": feedback}]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    compact = context["events"][0]["feedback"]
    for key, full in answer.items():
        assert _read_context_reference(tmp_path, compact[key]) == full
    pool = compact["candidates"]
    assert _read_context_reference(tmp_path, pool) == candidates
    assert pool["preview"] == [{k: v for k, v in row.items() if k != "program"} for row in candidates]
    assert _read_context_reference(tmp_path, compact["hypotheses"]) == hypotheses
    fallback = compact["operational_fallback"]
    assert _read_context_reference(tmp_path, fallback["hypotheses"]) == hypotheses
    assert {k: v for k, v in fallback.items() if k != "hypotheses"} == {
        k: v for k, v in feedback["operational_fallback"].items() if k != "hypotheses"}
    assert compact["premise_design_action"] == feedback["premise_design_action"]
    assert request == original
    assert json.loads((tmp_path / "research_context.json").read_text()) == original


@pytest.mark.parametrize("focus_changed", [False, True])
def test_unchanged_past_families_share_full_reference_but_new_verdicts_and_focus_changes_remain(
        tmp_path, focus_changed):
    family = {"id": "authored_cubic", "status": "SAT_WITNESS", "reason": "compatible",
              "basis": [{"op": "feature", "name": "signal"}]}
    questions = [{"id": f"response_{i}", "question": "Which response mechanism survives?",
        "status": "inconclusive", "evidence": [20, 30 + i], "answer": {
            "family_research": {"response_count": 31 + i, "no_progress_cycles": i,
                "selected_role": "shared_premise_research",
                "research_obligation": {"question": f"Research obligation {i}"},
                "families": [deepcopy(family)]}}} for i in range(4)]
    questions[2]["answer"]["family_research"]["families"][0].update(
        status="UNSAT_VERIFIED", reason="measured_counterexample")
    decision = {"questions": questions}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    focus = deepcopy(questions[1])
    if focus_changed:
        focus["answer"]["family_research"]["research_obligation"]["question"] = "New unsaved premise test"
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "files": {"method_decision.json": json.dumps(decision)}, "research_focus": {"question": focus}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    shown = context["method_decision"]["questions"]
    repeated = shown[1]["answer"]["family_research"]
    assert _read_context_reference(tmp_path, repeated["families"]) == [family]
    assert repeated["families"]["json_path"] == ["questions", 0, "answer", "family_research", "families"]
    for key in ("response_count", "no_progress_cycles", "selected_role", "research_obligation"):
        assert repeated[key] == questions[1]["answer"]["family_research"][key]
    changed = shown[2]["answer"]["family_research"]["families"][0]
    assert changed["status"] == "UNSAT_VERIFIED"
    assert changed["reason"] == "measured_counterexample"
    assert _read_context_reference(tmp_path, changed["basis"]) == family["basis"]
    assert shown[3] == questions[3]
    shown_focus = context["research_focus"]["question"]
    if focus_changed:
        assert shown_focus == focus
    else:
        assert _read_context_reference(tmp_path, shown_focus) == questions[1]
        assert shown_focus["id"] == focus["id"]
        assert shown_focus["status"] == focus["status"]
    assert request == original


def test_program_repairs_share_unchanged_answer_without_merging_changed_question_or_new_answer(tmp_path):
    answer = {"family_research": {"status": "shared_premise_research",
                                  "diagnostic": "complete retained result " * 700}}
    current = {"id": "response_15", "status": "inconclusive", "question": "Revised question",
               "answer": deepcopy(answer), "evidence": [21, 30]}
    previous = {**deepcopy(current), "question": "Original question"}
    changed = deepcopy(current)
    changed["answer"]["family_research"]["status"] = "new_contradiction"
    decision = {"questions": [current]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "files": {"method_decision.json": json.dumps(decision)}, "events": [
            {"feedback": {"question": previous, "procedure": {"error": "parse_error"}}},
            {"feedback": {"question": deepcopy(current), "procedure": {"status": "repaired"}}},
            {"feedback": {"question": changed, "procedure": {"status": "new_evidence"}}}]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    for index in (0, 1):
        feedback = context["events"][index]["feedback"]
        assert _read_context_reference(tmp_path, feedback["question"]["answer"]) == answer
        assert {k: v for k, v in feedback["question"].items() if k != "answer"} == {
            k: v for k, v in request["events"][index]["feedback"]["question"].items() if k != "answer"}
        assert feedback["procedure"] == request["events"][index]["feedback"]["procedure"]
    assert context["events"][2]["feedback"] == request["events"][2]["feedback"]
    assert request == original


def test_equal_experiment_prediction_vectors_are_shared_but_changed_outcomes_stay_inline(tmp_path):
    predictions = {f"response_{i}": False for i in range(30)}
    changed = {**predictions, "response_7": True}
    experiments = [{"id": f"setting_{i}", "number": 5, "prediction": 200. + i,
                    "predictions": deepcopy(vector)}
                   for i, vector in enumerate((predictions, predictions, changed, changed))]
    small = {"response_0": False}
    numerical = {f"response_{i}": float(i) for i in range(30)}
    request = {"evidence": {"observations_file": "laboratory_observations.json"}, "events": [
        {"feedback": {"operator": "reference_response", "experiments": experiments,
                      "small_predictions": {"predictions": small},
                      "numerical_predictions": {"predictions": numerical}}}]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    feedback = context["events"][0]["feedback"]
    shown = feedback["experiments"]
    assert shown[0]["predictions"] == predictions
    assert shown[2]["predictions"] == changed
    for index in (1, 3):
        assert _read_context_reference(tmp_path, shown[index]["predictions"]) == experiments[index]["predictions"]
    for actual, expected in zip(shown, experiments):
        assert {k: v for k, v in actual.items() if k != "predictions"} == {
            k: v for k, v in expected.items() if k != "predictions"}
    assert feedback["small_predictions"]["predictions"] == small
    assert feedback["numerical_predictions"]["predictions"] == numerical
    assert request == original


def test_source_and_decision_comparisons_share_pool_after_canonical_family_references(tmp_path):
    candidates = [{"id": f"member_{index}", "family": "conditional", "origin": "authored",
        "authored_id": "regime_rule", "status": "rejected", "target_prediction": 577. + index,
        "conflicting_responses": [{"crystal": 5, "frequency": 577., "activated": False,
                                    "evidence": [510]}],
        "program": {"op": "add", "children": [{"op": "const", "value": j}
                                                for j in range(100)]}}
        for index in range(20)]
    family = {"families": [{"id": "conditional", "status": "SAT_WITNESS"}],
              "selected_role": "shared_premise_research"}
    comparison = {"operator": "reference_response", "question_id": "response",
        "candidates": candidates, "family_research": family,
        "new_responses": [{"crystal": 5, "frequency": 577., "activated": False, "evidence": [510]}],
        "revision_action": {"action": "run", "arguments": {"operator": "propose_explanations",
                                                            "path": "shared/response.py"}}}
    decision = {"questions": [{"id": "response", "status": "inconclusive",
                               "answer": {"family_research": deepcopy(family)}}],
                "explanation_proposals": [{"comparison_result": deepcopy(comparison)}]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "files": {"method_decision.json": json.dumps(decision)},
        "research": {"llm_inquiries": {"source": {"comparison_result": comparison}}}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    source = context['research']['llm_inquiries']['source']['comparison_result']
    repeated = context['method_decision']['explanation_proposals'][0]['comparison_result']
    assert _read_context_reference(tmp_path, source['family_research']) == family
    assert 'candidates' not in repeated
    assert _read_context_reference(tmp_path, repeated) == comparison
    assert repeated['new_responses'] == comparison['new_responses']
    assert repeated['revision_action'] == comparison['revision_action']
    for row in source['candidates']['preview']:
        assert row['origin'] == 'authored' and row['authored_id'] == 'regime_rule'
        assert row['status'] == 'rejected' and row['conflicting_responses'][0]['activated'] is False
    assert request == original


def test_changed_response_comparisons_share_only_identical_pools_and_counterexamples(tmp_path):
    conflicts = [{"crystal": 5, "frequency": 570. + i, "activated": False, "evidence": [510 + i]}
                 for i in range(8)]
    candidates = [{"id": f"member_{index}", "family": "conditional", "origin": "authored",
        "status": "rejected", "target_prediction": 577. + index, "response_count": 332,
        "conflicting_responses": deepcopy(conflicts),
        "program": {"op": "add", "children": [{"op": "const", "value": j}
                                                for j in range(100)]}}
        for index in range(20)]
    original = {"operator": "reference_response", "candidates": candidates, "response_count": 332,
                "family_research": {}, "experiment_action": None}
    same_pool = {**deepcopy(original), "response_count": 333,
        "experiment_action": {"action": "run", "arguments": {"experiment": {"setting": 612.}}}}
    changed = deepcopy(same_pool)
    changed['candidates'][0]['response_count'] = 333
    changed['candidates'][1]['conflicting_responses'].append(
        {"crystal": 5, "frequency": 612., "activated": False, "evidence": [530]})
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
        "research": {"llm_inquiries": {name: {"comparison_result": value}
                     for name, value in [('first', original), ('same', same_pool), ('changed', changed)]}}}
    before = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    first, same, revised = [context['research']['llm_inquiries'][name]['comparison_result']
                            for name in ('first', 'same', 'changed')]
    assert first['candidates']['preview'][0]['conflicting_responses'] == conflicts
    assert 'preview' not in same['candidates']
    assert _read_context_reference(tmp_path, same['candidates']) == candidates
    assert same['response_count'] == 333
    assert same['experiment_action'] == same_pool['experiment_action']
    assert _read_context_reference(tmp_path, revised['candidates']) == changed['candidates']
    reused = revised['candidates']['preview'][0]['conflicting_responses']
    assert _read_context_reference(tmp_path, reused) == conflicts
    assert revised['candidates']['preview'][1]['conflicting_responses'] == changed['candidates'][1]['conflicting_responses']
    assert revised['experiment_action'] == changed['experiment_action']
    assert request == before


@pytest.mark.parametrize("answer", [
    "The latest target trial was negative; no activating setting is established.",
    {"family_research": {"status": "SAT_WITNESS"}},
])
def test_text_analysis_answer_keeps_prior_structured_response_feedback(tmp_path, answer):
    family = {"status": "SAT_WITNESS"}
    feedback = {"operator": "reference_response", "question_id": "extension",
                "family_research": deepcopy(family)}
    decision = {"questions": [{"id": "extension", "status": "inconclusive",
                               "answer": deepcopy(answer)}]}
    (tmp_path / "method_decision.json").write_text(json.dumps(decision))
    request = {"evidence": {"observations_file": "laboratory_observations.json"},
               "files": {"method_decision.json": json.dumps(decision)},
               "events": [{"action": "run", "feedback": feedback}]}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions="test")

    assert context["method_decision"]["questions"][0]["answer"] == answer
    retained = context["events"][0]["feedback"]["family_research"]
    if isinstance(answer, dict):
        assert _read_context_reference(tmp_path, retained) == family
    else:
        assert retained == family
    assert request == original
