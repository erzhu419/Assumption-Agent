"""The matched runner must measure model adoption, not pre-execute the archive."""

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from scripts import run_response_matched_adoption as runner
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_discoveryworld_method_reuse import METHOD_PATH, ReferenceLab
from test_discoveryworld_method_transfer import source_bundle
from test_world_reference_hypothesis import ReferenceLab as CalibrationLab


def run_scripted(tmp_path, monkeypatch, arm, *, adopt):
    agents, first_decisions = [], []
    lab = ReferenceLab()
    bundle = source_bundle()
    before = deepcopy(bundle)
    steps = []
    if adopt:
        steps.append(("run", {"operator": "procedure", "method_id": "reference"}))
    steps.append(("finish", {"lesson": "No additional experiment requested"}))

    class Provider(ScriptedProvider):
        def next_action(self, request):
            if not self.requests:
                first_decisions.append({"calls": agents[0].costs["model_calls"],
                    "actions": deepcopy(lab.actions), "observations": deepcopy(agents[0].observations),
                    "method_results": deepcopy(agents[0].method_results)})
            reply = super().next_action(request)
            reply["usage"] = ({"input_tokens": 20, "output_tokens": 3,
                "cached_input_tokens": 8, "total_tokens": 23} if len(self.requests) == 1
                else {"input_tokens": 17, "output_tokens": 2, "total_tokens": 19})
            return reply

    provider = Provider(steps)

    def make_agent(*args, **kwargs):
        agent = DiscoveryWorldAgent(*args, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(runner.study, "TransferSignalLab", lambda case: lab)
    monkeypatch.setattr(runner.study, "CASES", deepcopy(runner.study.CASES))
    monkeypatch.setattr(runner, "DiscoveryWorldAgent", make_agent)
    row = runner.run_case("toy", {}, 1, arm, bundle, tmp_path, LIMITS,
                          provider_factory=lambda workspace: provider)
    assert bundle == before
    assert len(agents) == 1 and agents[0].agent_arm == arm
    assert first_decisions == [{"calls": 1, "actions": [], "observations": [], "method_results": []}]
    assert provider.requests[0]["files"][METHOD_PATH] == bundle["files"][METHOD_PATH]
    assert provider.requests[0]["evidence"]["observation_count"] == 0
    result_path = tmp_path / "toy" / "repeat_01" / arm / "result.json"
    return row, json.loads(result_path.read_text()), agents[0], lab


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_finish_without_model_adoption_does_not_run_the_shared_controller(tmp_path, monkeypatch, arm):
    row, result, agent, lab = run_scripted(tmp_path, monkeypatch, arm, adopt=False)
    assert lab.actions == [] and agent.observations == []
    assert result["agent_arm"] == arm
    assert row["success"] is False
    assert row["supplied_method_selected"] is False
    assert row["supplied_method_executed"] is False
    assert row["point_confirmations"] == []
    assert row["costs"]["model_calls"] == 1
    assert row["costs"]["environment_steps"] == 0


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_model_can_execute_the_same_supplied_method_and_preserves_cost_missingness(tmp_path, monkeypatch, arm):
    row, result, agent, lab = run_scripted(tmp_path, monkeypatch, arm, adopt=True)
    assert lab.actions == [{"action": "TELEPORT_TO_OBJECT", "arg1": 7},
        {"action": "TALK", "arg1": 7}, {"chosen_dialog_option_int": 4},
        {"chosen_dialog_option_int": 2}]
    assert row["supplied_method_selected"] is True
    assert row["supplied_method_executed"] is True
    assert row["source_changed"] is False
    assert agent._provenance(METHOD_PATH) == {
        "inherited_program": True, "inherited_source_unchanged": True}
    assert row["success"] is False and row["point_confirmations"] == []
    assert result["events"][0]["arguments"]["method_id"] == "reference"
    assert row["costs"]["model_calls"] == 2
    assert row["costs"]["reported_tokens"] == 42
    assert row["costs"]["cached_input_tokens"] == 8
    assert row["costs"]["environment_steps"] == 4
    assert row["cache"] == {"reported_calls": 1, "missing_calls": 1,
                            "hit_calls": 1, "zero_hit_calls": 0}
    summary = runner.summarize([row], planned_runs=18)
    assert summary["status"] == "running" and summary["finished_runs"] == 1
    totals = summary["arms"][arm]
    assert totals["method_executed_runs"] == 1 and totals["successful_runs"] == 0
    assert totals["target_point_confirmations"] == 0
    assert totals["cache"] == row["cache"]
    for key, value in totals["costs"].items():
        assert value == row["costs"][key]


def point_result(*, status="supported", actual=100., pending=False):
    lab = CalibrationLab()
    lab.installed.add(5)
    lab.required[5] = lab.frequencies[5] = actual
    record = {"index": 2, **lab.act({"action": "TALK", "arg1": 105})}
    trial = {"question": "prediction_5", "number": 5, "prediction": 100.,
             "family": "candidate", "crystal": 205, "reactor": 105, "started": 2,
             "status": status}
    return {"reference_state": {"target_numbers": [4, 5],
                "tests": [] if pending else [trial], "test": trial if pending else None},
            "research": {"questions": [{"id": "prediction_5", "status": status,
                "answer": {"prediction": 100., "actual_frequency": actual, "activated": True},
                "evidence": [2]}]}, "observations": [record]}


@pytest.mark.parametrize("status,actual,confirmed", [
    ("supported", 100., True), ("proposed", 100., False),
    ("inconclusive", 100., False), ("refuted", 100., False),
    ("supported", 99., False)])
def test_requested_point_confirmation_requires_the_actual_requested_point(status, actual, confirmed):
    rows = runner.point_confirmations(point_result(status=status, actual=actual))
    assert len(rows) == 1
    assert rows[0]["physically_confirmed"] is confirmed


def test_terminal_pending_point_is_counted_but_reference_restoration_is_not():
    result = point_result(pending=True)
    assert runner.point_confirmations(result)[0]["physically_confirmed"] is True
    result["reference_state"]["test"]["restore_frequency"] = 100.
    assert runner.point_confirmations(result) == []


def test_native_research_assignment_names_the_executed_parent_and_current_evidence():
    failure = "shared/reference_extension_5.json"
    question = runner.recipe.reference_research_question(5, failure, "shared/alternate_research.py")
    assert "operator:'reference_hypothesis',path:'shared/alternate_research.py'" in question
    assert "Choose the basis and revise it using physical counterexamples" in question
    assert failure + " is the historical snapshot" in question
    assert "shared/reference_recalibration.json contains the latest" in question
    # The two-argument experiment-script entry retains its actual conventional path.
    assert "path:'shared/consolidated_method.py'" in runner.recipe.reference_research_question(5, failure)


@pytest.mark.parametrize('helper_present', (False, True))
def test_prepared_bundle_enters_live_extension_without_losing_its_helper(tmp_path, monkeypatch, helper_present):
    method_path = "shared/alternate_research.py"
    monkeypatch.setattr(runner.study, "METHOD_PATH", method_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shared").mkdir()
    current = Path(runner.recipe.__file__).read_text()
    functions = {node.name: ast.get_source_segment(current, node)
                 for node in ast.parse(current).body if isinstance(node, ast.FunctionDef)}
    declarations = [{"id": f"retained_{index}", "basis": [{"op": "feature", "name": "density"}]}
                    for index in range(16)]
    source = ("import json\nfrom pathlib import Path\n"
              "STATE = Path('shared/reference_recalibration.json')\n"
              + "ESTIMATORS = " + repr(declarations) + "\n\n"
              + functions["_save"] + "\n\n" + functions["_extend_hypotheses"] + "\n")
    # Older exports may already contain the broad assignment and its two-argument call.
    source = source.replace("reference_research_question(number, failure_file, path)",
                            "reference_research_question(number, failure_file)")
    if helper_present:
        source += "\ndef reference_research_question(number, failure_file):\n    return 'Old broad assignment'\n"
    path = tmp_path / method_path
    path.write_text(source)
    original = {"files": {method_path: source}, "methods": []}
    context = {"number": 4, "estimators": declarations, "eligible_targets": [],
               "control_grids": {}, "responses": [], "measurements": []}
    state = {"completed": [], "rejected": {}, "response_context": context,
             "response_research": {"status": "family_research_needed"}}
    memory = {"model_available": True, "limits": {"max_model_calls": 4},
              "costs": {"model_calls": 0}, "llm_inquiries": {}}
    missing = {}
    exec(compile(source, method_path, "exec"), missing)
    if helper_present:
        assert missing["_extend_hypotheses"](deepcopy(state), memory, 4, method_path)[
            "arguments"]["question"]["question"] == "Old broad assignment"
    else:
        with pytest.raises(NameError, match="reference_research_question"):
            missing["_extend_hypotheses"](deepcopy(state), memory, 4, method_path)
    prepared = runner.prepare_bundle(original)
    assert original["files"][method_path] == source
    assert prepared["methods"] == original["methods"]
    completed = prepared["files"][method_path]
    other = lambda text: [ast.dump(node) for node in ast.parse(text).body
        if not (isinstance(node, ast.FunctionDef) and node.name == "reference_research_question")]
    assert other(completed) == other(source.replace(
        "reference_research_question(number, failure_file)",
        "reference_research_question(number, failure_file, path)"))
    path.write_text(completed)
    exported = {}
    exec(compile(completed, method_path, "exec"), exported)
    assert exported["ESTIMATORS"] == declarations
    result = exported["_extend_hypotheses"](deepcopy(state), memory, 4, method_path)
    assert result["arguments"]["operator"] == "inquire"
    assert result["arguments"]["question"]["id"] == "reference_extension_4"
    question = result["arguments"]["question"]["question"]
    assert "shared/reference_extension_4.json" in question
    assert "operator:'reference_hypothesis',path:'shared/alternate_research.py'" in question
    assert "shared/reference_recalibration.json contains the latest" in question


def legacy_response_source(source):
    """The actual archived omissions, retaining the rest of the controller."""
    tree = ast.parse(source)
    route = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "_point_needs_response")
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "estimator")
    position = next(i for i, node in enumerate(loop.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == "paid_check" for target in node.targets))
    old_route = '''def _point_needs_response(state, number):
    supported = {test['family'] for test in state.get('tests', [])
                 if test['status'] == 'supported' and test['number'] != number
                 and test['number'] in state.get('target_numbers', [])}
    return any(test['number'] == number and test['status'] == 'refuted'
               and not test.get('response_question') and test['family'] in supported
               for test in state.get('tests', []))
'''
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted([
            (route.lineno - 1, route.end_lineno, old_route),
            (loop.body[position].lineno - 1, loop.body[position + 1].end_lineno, "")], reverse=True):
        lines[start:end] = [replacement]
    return "".join(lines).replace(
        "def select_actions(public_spec, observations, memory):\n",
        "def select_actions(public_spec, observations, memory):\n"
        "    if memory.get('author_choice'):\n"
        "        return {'stop': 'Author retained this branch'}\n", 1)


def test_preparation_restores_completed_target_routing_without_replacing_author_code():
    source = legacy_response_source(Path(runner.recipe.__file__).read_text())
    original = {"files": {runner.study.METHOD_PATH: source}, "methods": []}
    prepared = runner.prepare_bundle(original)
    assert original["files"][runner.study.METHOD_PATH] == source
    assert runner.prepare_bundle(prepared) == prepared
    revised = prepared["files"][runner.study.METHOD_PATH]

    def unaffected_ast(text):
        tree = ast.parse(text)
        tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef)
            and node.name in ("_extend_hypotheses", "_point_needs_response", "reference_research_question"))]
        for loop in ast.walk(tree):
            if not (isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)
                    and loop.target.id == "estimator"):
                continue
            index = next((i for i, node in enumerate(loop.body)
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == "paid_check" for target in node.targets)), None)
            if index is not None:
                del loop.body[index:index + 2]
        return ast.dump(tree)

    assert unaffected_ast(revised) == unaffected_ast(source)
    before, after = {}, {}
    exec(compile(source, "old_export.py", "exec"), before)
    exec(compile(revised, "prepared_export.py", "exec"), after)
    assert before["ESTIMATORS"] == after["ESTIMATORS"]
    assert after["select_actions"]({}, [], {"author_choice": True}) == {
        "stop": "Author retained this branch"}
    state = {"target_numbers": [4, 5], "completed": [4], "tests": [
        {"number": 4, "family": "old_shape", "status": "refuted"},
        {"number": 5, "family": "new_shape", "status": "refuted"}]}
    assert not before["_point_needs_response"](state, 5)
    assert after["_point_needs_response"](state, 5)
    state["completed"] = []
    state["tests"][0].update(status="supported", response_question="response_4")
    assert after["_point_needs_response"](state, 5)
    for change in ({"authored_experiment": True}, {"restore_frequency": 100.}):
        changed = deepcopy(state)
        changed["tests"][0].update(change)
        assert not after["_point_needs_response"](changed, 5)


@pytest.mark.parametrize("change", ("unchanged", "unknown", "basis", "measurement"))
def test_prepared_controller_consumes_paid_exclusions_and_reopens_changed_bindings(tmp_path, change):
    from test_world_reference_hypothesis import PATH, _paid_shared_band_conflict, square, submit

    agent, lab = _paid_shared_band_conflict(tmp_path)
    hypothesis = square("density", "density_square")
    result = submit(agent, lab, hypothesis)
    assert result["paid_family_check"]["status"] == "UNSAT_VERIFIED"
    state_path = agent.workspace / runner.recipe.STATE
    state = json.loads(state_path.read_text())
    source = legacy_response_source((agent.workspace / PATH).read_text())
    original = {"files": {runner.study.METHOD_PATH: source}, "methods": []}
    revised = runner.prepare_bundle(original)["files"][runner.study.METHOD_PATH]

    def controller(text):
        exported = {}
        exec(compile(text, PATH, "exec"), exported)
        exported.update(STATE=state_path, FAMILIES=(), ESTIMATORS=(hypothesis,),
            _extend_hypotheses=lambda *args: {"stop": "No surviving point family"})
        return exported["select_actions"]

    memory = {"program_path": PATH, "questions": agent.questions}
    spec = {"current_observation": lab.public_observation()}
    # Reproduce the old export's actual mistake: it fits the paid-refuted family.
    old_action = controller(source)(spec, agent.observations, memory)
    assert old_action["arguments"]["operator"] == "research_update"
    assert old_action["arguments"]["question"]["id"] == "reference_fit_density_square_4"
    if change == "unknown":
        state["family_history_checks"][hypothesis["id"]]["status"] = "UNKNOWN"
    elif change == "basis":
        hypothesis = square("gap", "density_square")
    elif change == "measurement":
        lab.density[2] = 2.
        agent._execute("query", {"actions": [{"action": "USE", "arg1": 91, "arg2": 202}]}, lab)
        spec = {"current_observation": lab.public_observation()}
    state_path.write_text(json.dumps(state))
    action = controller(revised)(spec, agent.observations, memory)
    if change == "unchanged":
        assert action == {"stop": "No surviving point family"}
    else:
        assert action["arguments"]["operator"] == "research_update"
        assert action["arguments"]["question"]["id"] == "reference_fit_density_square_4"
    assert agent.costs["model_calls"] == 0


def scientific_result():
    result = point_result()
    trial = result["reference_state"]["tests"][0]
    trial.update(authored_experiment=True, parent_id="extension")
    result["research"]["questions"].append({"id": "extension", "parent_id": "task",
        "reviewed_child_answers": {"prediction_5": {"status": "supported", "evidence": [2]}}})
    result["research"]["llm_inquiries"] = {"extension": {"status": "answered"}}
    result["events"] = [{"action": "run", "usage": {}, "model_call": 3,
        "inquiry_question_id": "extension", "arguments": {"operator": "reference_response",
            "experiment": {"number": 5, "setting": 100.}},
        "feedback": {"question_id": "prediction_5", "new_observation_indices": [2]}}]
    result["method_results"] = [
        {"step": 7, "path": "method.py", "proposal": {"action": "run", "arguments": {
            "operator": "inquire", "question": {"id": "extension"}}}},
        {"step": 8, "path": "other.py", "proposal": {"actions": [{"action": "wrong_parent"}]}},
        {"step": 9, "path": "method.py", "proposal": {"action": "run", "arguments": {
            "operator": "reference_response", "experiment": {"number": 4, "setting": 150.}}}}]
    return result


def test_scientific_trace_links_child_measurement_to_review_and_actual_parent_continuation():
    result = scientific_result()
    before = deepcopy(result)
    trace = runner.scientific_trace(result)
    assert result == before
    decision = trace["decisions"][0]
    assert decision["inquiry_question_id"] == "extension" and decision["model_call"] == 3
    assert decision["fresh_observations"] == [2]
    trial = decision["physical_trials"][0]
    assert trial["fresh_evidence"] == [2] and trial["reviewed_by"] == ["extension"]
    assert trial["physically_confirmed"] is True
    continuation = trace["inquiries"][0]["parent_returns"][0]
    assert continuation["returned_at_program_step"] == 7
    assert continuation["parent_next_step"] == 9
    assert continuation["parent_next_experiment"] == {"number": 4, "setting": 150.}
    assert trace["confirmed_decision_trials"] == 1


@pytest.mark.parametrize("fresh,status", [([], "supported"), ([999], "supported"), ([2], "refuted")])
def test_scientific_trace_does_not_promote_stale_missing_or_refuted_evidence(fresh, status):
    result = scientific_result()
    result["events"][0]["feedback"]["new_observation_indices"] = fresh
    result["research"]["questions"][0]["status"] = status
    result["method_results"][-1]["proposal"] = {"stop": "Further evidence needed"}
    trace = runner.scientific_trace(result)
    assert trace["confirmed_decision_trials"] == 0
    assert trace["inquiries"][0]["parent_returns"][0]["parent_stop"] == "Further evidence needed"
    if fresh != [2]:
        assert trace["decisions"][0]["physical_trials"][0]["reviewed_by"] == []


def test_scientific_trace_captures_model_declaration_but_not_source_edits_or_host_scheduling():
    result = scientific_result()
    hypothesis = {"id": "new_basis", "basis": [{"op": "feature", "name": "signal"}]}
    result["events"][0]["arguments"] = {"operator": "reference_hypothesis", "hypothesis": hypothesis}
    result["events"][0]["feedback"] = {"hypothesis": hypothesis,
        "pending_test": deepcopy(result["reference_state"]["tests"][0]),
        "new_observation_indices": [2]}
    trace = runner.scientific_trace(result)
    assert trace["decisions"][0]["declared_hypothesis"] == hypothesis
    assert trace["confirmed_decision_trials"] == 1
    result["events"][0].pop("usage")
    result["source_changed"] = True
    result["research"]["child_answers_returned"] = 50
    trace = runner.scientific_trace(result)
    assert trace["decisions"] == [] and trace["confirmed_decision_trials"] == 0
    assert trace["inquiries"][0]["model_events"] == 0


def test_scientific_trace_accepts_controller_action_lists_before_and_after_child_return():
    result = scientific_result()
    result["method_results"].insert(0, {"step": 6, "path": "method.py",
        "proposal": [{"action": "TALK", "arg1": 105}]})
    action = {"action": "USE", "arg1": 93, "arg2": 204}
    result["method_results"][-1]["proposal"] = [action]
    trace = runner.scientific_trace(result)
    returned = trace["inquiries"][0]["parent_returns"][0]
    assert returned["parent_next_step"] == 9
    assert returned["parent_next_world_actions"] == [action]
    assert trace["confirmed_decision_trials"] == 1
