"""Saved experiments remain executable options; only execution changes support."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import InstrumentLab, LIMITS, ScriptedProvider
from test_discoveryworld_analysis_continuation import make_agent
from test_discoveryworld_contrast import PATH, QUESTION, SOURCE, contrast, prepared, question


LOAD_MARKER = "from pathlib import Path\nPath('design_loaded.txt').write_text('executed')\n"


def decision_entry(agent, qid="physical"):
    request = agent._request()
    context = prepare_code_context(agent.workspace, request, schema={}, instructions="")
    entries = context["method_decision"]["questions"]
    return next(row for row in entries if row["id"] == qid)


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_saving_returns_native_execution_and_both_contexts_offer_it_without_running(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm=arm, outcome=True)
    before_executions = agent.program_executions
    agent.limits = {**agent.limits, "max_model_calls": 1}
    agent.provider = ScriptedProvider([("write", {"path": PATH, "code": LOAD_MARKER + SOURCE,
                                                   "question": QUESTION})])
    result = agent.run(lab)

    assert result["status"] == "model_call_budget_exhausted"
    saved = result["events"][0]["feedback"]
    action = saved["contrast_action"]
    assert saved["written"] == PATH
    assert action["action"] == "run"
    assert action["arguments"]["operator"] == "contrast" and action["arguments"]["path"] == PATH
    assert action["arguments"]["question"]["id"] == QUESTION["id"]
    assert action["arguments"]["question"] == {"id": QUESTION["id"]}
    assert agent.program_executions == before_executions and lab.environment_steps == 1
    assert not (tmp_path / "design_loaded.txt").exists()
    assert "answer" not in question(agent, "physical")
    assert not agent._state()["experiment_trials"]
    assert action in decision_entry(agent)["contrast_actions"]
    assert agent.program_executions == before_executions
    assert not (tmp_path / "design_loaded.txt").exists()

    executed = agent._execute(action["action"], action["arguments"], lab)
    assert executed["selected_experiment"]["id"] == "physical"
    assert executed["experiment"]["observation_indices"] == [1]
    assert question(agent, "physical")["status"] == "supported"
    assert (tmp_path / "design_loaded.txt").read_text() == "executed"
    assert lab.actions[-1] == {"action": "USE", "arg1": 8, "arg2": 7}


@pytest.mark.parametrize("association", ["question", "research_question"])
def test_inherited_successful_write_recovers_option_from_current_source_without_new_metadata(tmp_path, association):
    original, lab = prepared(tmp_path / "original", arm="code_archive")
    original._question(QUESTION)
    inherited_source = LOAD_MARKER + SOURCE
    restored = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path / "restored", arm="code_archive", limits=LIMITS)
    restored.lab = lab
    restored.questions = deepcopy(original.questions)
    restored.observations = deepcopy(original.observations)
    destination = restored.workspace / PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(inherited_source)
    # This is the already-supported old write-event schema, before a save
    # returned contrast_action. The source is read from its current file.
    restored.events = [{"action": "write", "arguments": {
        "path": PATH, "code": "# older recorded source", association: QUESTION},
        "feedback": {"written": PATH}}]
    before = deepcopy(restored.questions)

    options = decision_entry(restored)["contrast_actions"]

    assert len(options) == 1
    assert options[0]["arguments"]["path"] == PATH
    assert options[0]["arguments"]["question"]["id"] == QUESTION["id"]
    assert restored.questions == before and restored.program_executions == 0
    assert not (restored.workspace / "design_loaded.txt").exists()
    assert not restored._state()["experiment_trials"]


@pytest.mark.parametrize("source", [
    "def answer(spec, observations, memory):\n    return {}\n",
    "def factory():\n    def design(spec, observations, memory): return {}\n"
    "    def answer(spec, observations, memory): return {}\n",
])
def test_ordinary_non_design_file_has_no_contrast_execution_option(tmp_path, source):
    agent, lab = prepared(tmp_path, arm="code_archive")
    saved = agent._execute("write", {"path": PATH, "code": source, "question": QUESTION}, lab)
    assert saved["written"] == PATH and "contrast_action" not in saved
    assert lab.environment_steps == 1 and "answer" not in question(agent, "physical")


@pytest.mark.parametrize("new_draft", [False, True])
def test_offering_existing_or_new_design_preserves_answers_and_repaired_analysis(tmp_path, new_draft):
    agent, lab = prepared(tmp_path, arm="code_archive", outcome=True)
    completed = contrast(agent, lab)
    agent.events.append({"action": "run", "arguments": {"operator": "contrast", "path": PATH},
                         "feedback": deepcopy(completed)})
    repair = """def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': 'Repaired analysis retains a missing condition',
            'evidence': [observations[-1]['index']]}
"""
    agent._execute("evaluate", {"path": "shared/repaired_answer.py", "code": repair,
                                "question_id": "physical"}, lab)
    child_before = deepcopy(question(agent, "physical"))
    trials_before = deepcopy(agent._state()["experiment_trials"])
    executions_before = agent.program_executions
    if new_draft:
        selected_path = "shared/revised_design.py"
        saved = agent._execute("write", {"path": selected_path, "code": SOURCE,
                                         "question": {"id": "physical"}}, lab)
        action = saved["contrast_action"]
    else:
        selected_path = PATH
        options = decision_entry(agent)["contrast_actions"]
        action = next(action for action in options if action["arguments"]["path"] == PATH)

    assert action["arguments"]["question"] == {"id": "physical"}

    assert question(agent, "physical") == child_before
    assert agent._state()["experiment_trials"] == trials_before
    assert agent.program_executions == executions_before and lab.environment_steps == 2
    agent._execute(action["action"], action["arguments"], lab)
    updated = question(agent, "physical")
    assert updated["design_path"] == selected_path
    assert updated["status"] == "inconclusive"
    assert updated["analysis_path"] == "shared/repaired_answer.py"
    assert lab.environment_steps == 3


def test_saved_draft_without_prior_contrast_retains_later_paid_analysis_repair(tmp_path):
    agent, lab = prepared(tmp_path, arm="code_archive", outcome=True)
    saved = agent._execute("write", {"path": PATH, "code": SOURCE, "question": QUESTION}, lab)
    agent._execute("query", {"question": {"id": "physical"},
                              "actions": [{"action": "USE", "arg1": 8, "arg2": 7}]}, lab)
    repair = """def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': 'Observed activation; the additional condition remains unresolved',
            'evidence': [observations[-1]['index']]}
"""
    agent._execute("evaluate", {"path": "corrected.py", "code": repair, "question_id": "physical"}, lab)
    assert "design_path" not in question(agent, "physical")
    assert not agent._state()["experiment_trials"]

    action = saved["contrast_action"]
    assert action["arguments"]["question"] == {"id": "physical"}
    result = agent._execute(action["action"], action["arguments"], lab)

    assert result["selected_experiment"]["id"] == "physical"
    child = question(agent, "physical")
    assert child["analysis_path"] == "corrected.py" and child["status"] == "inconclusive"
    assert child["answer"].startswith("Observed activation;") and child["evidence"] == [2]
    assert lab.environment_steps == 3


@pytest.mark.parametrize("route", ["parent_program", "after_observation"])
def test_nested_actual_write_recovers_design_without_registering_unexecuted_proposal(tmp_path, route):
    agent, lab = prepared(tmp_path, arm="code_archive")
    agent.limits = {**agent.limits, "max_model_calls": 1}
    actual_write = {"action": "write", "arguments": {
        "path": PATH, "code": LOAD_MARKER + SOURCE, "question": QUESTION}}
    unused = "shared/unexecuted_design.py"
    (tmp_path / "shared").mkdir(exist_ok=True)
    (tmp_path / unused).write_text(SOURCE)
    proposed_only = {"action": "write", "arguments": {"path": unused, "question": QUESTION}, "written": unused}
    if route == "parent_program":
        proposal = {**actual_write, "planned_only": proposed_only}
        parent = ("def select_actions(spec, observations, memory):\n"
                  "    if memory['program_results']: return {'stop': 'Draft saved'}\n"
                  f"    return {proposal!r}\n")
        agent._execute("write", {"path": "shared/save_draft.py", "code": parent}, lab)
        steps = [("run", {"operator": "procedure", "path": "shared/save_draft.py", "max_steps": 3})]
    else:
        steps = [("query", {"actions": [{"action": "TALK", "arg1": 7}],
                             "after_observation": [actual_write], "planned_only": proposed_only})]
    agent.provider = ScriptedProvider(steps)
    result = agent.run(lab)
    assert result["status"] == "model_call_budget_exhausted"
    before = deepcopy(agent.questions)

    options = decision_entry(agent)["contrast_actions"]

    assert [action["arguments"]["path"] for action in options] == [PATH]
    assert agent.questions == before and "answer" not in question(agent, "physical")
    assert "design_path" not in question(agent, "physical")
    assert not (tmp_path / "design_loaded.txt").exists()
    assert not agent._state()["experiment_trials"]


def test_evaluating_old_paid_answer_keeps_another_childs_pending_analysis(tmp_path):
    agent, _, lab = make_agent(tmp_path, actual=False)
    old = {"id": "old", "question": "Does this candidate work?", "parent_id": "task"}
    agent._execute("query", {"question": old, "actions": [{"action": "OBSERVE"}]}, lab)
    initial = """def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': 'The previous analysis did not read the outcome', 'evidence': [0]}
"""
    agent._execute("evaluate", {"path": "old_answer.py", "code": initial, "question_id": "old"}, lab)
    agent._execute("query", {"question": {"id": "pending", "question": "What does this new setting do?",
        "parent_id": "task"}, "actions": [{"action": "SET"}]}, lab)
    first = deepcopy(agent._request()["research_focus"])
    assert first["kind"] == "analysis" and first["question"]["id"] == "pending"
    pending_before = deepcopy(question(agent, "pending"))
    revised = """def answer(spec, observations, memory):
    actual = observations[0]['observation']['active']
    return {'status': 'supported' if actual else 'refuted', 'answer': {'actual': actual}, 'evidence': [0]}
"""
    agent._execute("evaluate", {"path": "old_answer.py", "code": revised, "question_id": "old"}, lab)
    assert question(agent, "old")["status"] == "refuted" and question(agent, "old")["evidence"] == [0]
    assert question(agent, "pending") == pending_before
    assert agent._request()["research_focus"] == first
    assert lab.environment_steps == 2


@pytest.mark.parametrize("kind, allowed", [
    ("analysis", {"write", "run", "evaluate", "finish"}),
    ("operation", {"write", "run", "finish"}),
    ("intervention", {"write", "run", "finish"}),
])
def test_provider_schema_allows_evaluate_only_in_analysis_focus(tmp_path, monkeypatch, kind, allowed):
    def run(command, prompt, timeout, cwd):
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert set(schema["properties"]["action"]["enum"]) == allowed
        response = {"action": "evaluate" if kind == "analysis" else "finish",
                    "arguments_json": json.dumps({"path": "old_answer.py", "question_id": "old"})}
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(response))
        return 0, "", "", False

    monkeypatch.setattr(code_agent, "_run", run)
    provider = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test")
    result = provider.next_action({"task_interface": "Use public evidence.", "research_focus": {
        "kind": kind, "question": {"id": "pending", "question": "Read the pending evidence"},
        "path": "pending.py", "evidence": [1]}})
    assert result["action"] == ("evaluate" if kind == "analysis" else "finish")


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_authored_answer_exposes_only_its_cited_old_records_and_refreshes_after_explicit_repair(tmp_path, arm):
    from test_discoveryworld_analysis_repair import public_record
    from test_discoveryworld_prerequisite import ControlLab
    from test_discoveryworld_reference_history import record

    source = [public_record(index) for index in range(3)]
    for row in source:
        row["observation"]["lastActionMessage"] = "Set resonance frequency to 1296 Hz."
    source.extend(record(index, objects=[], action={"action": "MOVE_DIRECTION", "arg1": "north"})
                  for index in range(3, 20))
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm=arm, limits=LIMITS)
    lab = agent.lab = ControlLab()
    agent._question({"id": "task", "question": "Explain the physical response"})
    child = agent._question({"id": "activation", "question": "Does this setting activate the reactor?",
                             "parent_id": "task"})
    agent.observations = deepcopy(source[:2])
    child["acquired_observations"] = [1]
    agent._record_answer(child, {"status": "supported", "answer": "1296 Hz activated the reactor",
                                 "evidence": [1]})
    agent.observations.extend(deepcopy(source[2:]))
    child["acquired_observations"].extend([2, 19])
    original = deepcopy(child)
    executions = agent.program_executions

    context = prepare_code_context(tmp_path, agent._request(), schema={}, instructions="")
    entry = next(row for row in context["method_decision"]["questions"] if row["id"] == "activation")

    for field in ("recent_action_results", "reference_history", "measurement_history"):
        assert 1 not in {row["index"] for row in context["evidence"][field]}
    assert entry["answer_observations"] == [agent._action_observation(source[1])]
    assert "uncalibrated" in entry["answer_observations"][0]["public_relations"][0]["text"]
    assert entry["status"] == "supported" and entry["answer"] == original["answer"]
    assert entry["unreviewed_observation_indices"] == [2, 19]
    assert child == original and agent.observations == source
    assert agent.program_executions == executions and lab.environment_steps == 0

    repair = """def answer(spec, observations, memory):
    record = observations[2]
    state = record['public_relations'][0]['text']
    assert 'uncalibrated' in state
    return {'status': 'refuted', 'answer': state, 'evidence': [record['index']]}
"""
    agent._execute("evaluate", {"path": "corrected_activation.py", "code": repair,
                                "question_id": "activation"}, lab)
    executions = agent.program_executions
    refreshed = decision_entry(agent, "activation")
    assert refreshed["status"] == "refuted" and refreshed["evidence"] == [2]
    assert refreshed["answer_observations"] == [agent._action_observation(source[2])]
    assert agent.program_executions == executions and lab.environment_steps == 0
    assert agent.observations == source


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_inline_answer_repair_at_saved_design_path_preserves_design_for_corrected_rerun(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm=arm, outcome=False)
    saved = agent._execute("write", {"path": PATH, "code": LOAD_MARKER + SOURCE, "question": QUESTION}, lab)
    original_source = (tmp_path / PATH).read_bytes()
    agent._execute("query", {"question": {"id": "physical"},
                              "actions": [{"action": "USE", "arg1": 8, "arg2": 7}]}, lab)
    wrong = """def answer(spec, observations, memory):
    return {'status': 'supported', 'answer': 'The candidate activated the device', 'evidence': [1]}
"""
    agent._execute("evaluate", {"path": "wrong_answer.py", "code": wrong, "question_id": "physical"}, lab)
    assert "design_path" not in question(agent, "physical")
    assert not agent._state()["experiment_trials"]
    corrected = """def answer(spec, observations, memory):
    record = observations[-1]
    outcome = record['observation']['physical_outcome']
    return {'status': 'supported' if outcome else 'refuted',
            'answer': {'actual_outcome': outcome}, 'evidence': [record['index']]}
"""

    agent._execute("evaluate", {"path": PATH, "code": corrected, "question_id": "physical"}, lab)

    child = question(agent, "physical")
    assert child["status"] == "refuted" and child["evidence"] == [1]
    assert child["analysis_path"] == "shared/analysis_physical.py"
    assert (tmp_path / child["analysis_path"]).read_text() == corrected
    assert (tmp_path / PATH).read_bytes() == original_source
    assert lab.environment_steps == 2 and not (tmp_path / "design_loaded.txt").exists()
    action = saved["contrast_action"]
    result = agent._execute(action["action"], action["arguments"], lab)
    assert result["selected_experiment"]["id"] == "physical"
    assert child["analysis_path"] == "shared/analysis_physical.py"
    assert child["status"] == "refuted" and child["evidence"] == [2]
    assert child["answer"] == {"actual_outcome": False}
    assert (tmp_path / PATH).read_bytes() == original_source
    assert (tmp_path / "design_loaded.txt").read_text() == "executed"
    assert lab.environment_steps == 3


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_inline_child_answer_at_parent_procedure_path_preserves_executable_parent(tmp_path, arm):
    lab = InstrumentLab()
    agent = DiscoveryWorldAgent(None, tmp_path, arm=arm, limits=LIMITS)
    agent.lab = lab
    agent._question({"id": "task", "question": "Read the instrument and tune it"})
    agent._question({"id": "procedure_repair_1", "parent_id": "task",
                     "question": "What reading supports the next intervention?"})
    path = "shared/consolidated_method.py"
    source = LOAD_MARKER + """def select_actions(spec, observations, memory):
    child = memory['answers']['procedure_repair_1']
    assert child['status'] == 'supported'
    return {'actions': [{'action': 'TUNE', 'arg1': 2 * child['answer']['reading']}]}
"""
    agent._execute("write", {"path": path, "code": source}, lab)
    agent._execute("query", {"actions": [{"action": "MEASURE"}],
                             "question": {"id": "procedure_repair_1"}}, lab)
    analysis = """def answer(spec, observations, memory):
    record = observations[-1]
    return {'status': 'supported', 'answer': {'reading': record['feedback']['reading']},
            'evidence': [record['index']]}
"""

    evaluated = agent._execute("evaluate", {"path": path, "code": analysis,
                                            "question_id": "procedure_repair_1"}, lab)

    child = evaluated["question"]
    assert child["status"] == "supported" and child["evidence"] == [0]
    assert child["analysis_path"] == "shared/analysis_procedure_repair_1.py"
    assert (tmp_path / child["analysis_path"]).read_text() == analysis
    assert (tmp_path / path).read_text() == source
    assert lab.environment_steps == 1 and not (tmp_path / "design_loaded.txt").exists()

    continued = agent._run_procedure(lab, path, max_steps=2,
        query_arguments={"procedure_start": 0, "research_question": {"id": "task"}})
    assert continued["status"] == "completed"
    assert lab.actions == [{"action": "MEASURE"}, {"action": "TUNE", "arg1": 6}]
    assert lab.score()["completedSuccessfully"]
    assert (tmp_path / path).read_text() == source
    assert (tmp_path / "design_loaded.txt").read_text() == "executed"
