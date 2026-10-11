"""Exhausted reference hypotheses enter the existing child loop and return."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import InstrumentLab, LIMITS, ScriptedProvider


PATH = "shared/current_method.py"
QUESTION = "reference_extension_3"
FAILED = {
    "completed": [], "rejected": {"3": ["density", "gap"]},
    "tests": [{"number": 3, "family": "gap", "prediction": 2,
               "status": "refuted", "question": "reference_test_3_gap"}],
    "out_of_range": [{"number": 3, "family": "density", "prediction": -4}],
}


def wrapper():
    # Export the actual production recipe. Only bypass its world-specific
    # acquisition in this synthetic laboratory; the extension is unmodified.
    return Path(recipe.__file__).read_text() + """
def select_actions(public_spec, observations, memory):
    state = json.loads(STATE.read_text())
    return _extend_hypotheses(state, memory, 3, memory['program_path'])
"""


def test_declared_trial_enters_response_research_without_expanding_its_bases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    declarations = ({'id': 'measured_signal', 'basis': [{'op': 'feature', 'name': 'signal'}]},)
    monkeypatch.setattr(recipe, 'ESTIMATORS', declarations)
    state = deepcopy(FAILED)
    state['research_table'] = {'references': [{'crystal': 1, 'measurements': {
        'signal': {'value': 2., 'evidence': 1}}}], 'targets': []}
    state['response_settings'] = [{'crystal': 1, 'frequency': 3., 'activated': True}]
    result = recipe._extend_hypotheses(state, {
        'program_arguments': {'estimator_trial': True},
        'limits': {'max_model_calls': 4}, 'costs': {'model_calls': 0}}, 3, PATH)
    assert result['arguments']['operator'] == 'reference_response'
    context = result['arguments']['context']
    assert context['estimators'] == list(declarations)
    assert context['responses'] == [[1, 3., True]]
    assert not list(tmp_path.iterdir())


def test_declared_trial_returns_family_obligation_instead_of_offline_fallback_or_nested_inquiry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = deepcopy(FAILED)
    state['response_research'] = {'status': 'family_research_needed',
        'family_research': {'research_obligation': {
            'question': 'Which new measurement separates the contradictory responses?'}},
        'operational_fallback': {'selected_experiment': {'family': 'old', 'prediction': 7.}}}
    for model_available in (False, True):
        result = recipe._extend_hypotheses(state, {
            'program_arguments': {'estimator_trial': True},
            'model_available': model_available,
            'limits': {'max_model_calls': 4}, 'costs': {'model_calls': 0}}, 3, PATH)
        assert 'family_research_needed' in result['stop']
        assert 'Which new measurement' in result['stop']
        assert 'action' not in result and 'response_trial' not in result
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('model_available', [False, True])
def test_parent_consumes_available_composition_before_fallback_or_author(model_available):
    action = {'action': 'run', 'arguments': {
        'operator': 'reference_expansion', 'path': PATH}}
    state = deepcopy(FAILED)
    state['response_research'] = {'status': 'family_research_needed',
        'basis_expansion_action': action,
        'operational_fallback': {'selected_experiment': {
            'family': 'old', 'prediction': 7.}}}
    memory = {'model_available': model_available,
        'limits': {'max_model_calls': 4}, 'costs': {'model_calls': 0}}
    assert recipe._extend_hypotheses(state, memory, 3, PATH) == action
    # A trial of one authored declaration must return its own evidence rather
    # than silently introducing another scientific structure.
    trial = recipe._extend_hypotheses(state, {
        **memory, 'program_arguments': {'estimator_trial': True}}, 3, PATH)
    assert 'family_research_needed' in trial['stop']
    assert 'action' not in trial


def test_actual_offline_fallback_enters_existing_next_response_progress(tmp_path, monkeypatch):
    from test_world_response_consequences import fixture, assess, experiment, response

    monkeypatch.chdir(tmp_path)
    state = fixture(panel_crystal=4, panel_frequency=11.)
    state['expanded_features'] = ['x']
    first = assess(state, experiments=[experiment('negative_7', 1, 7.)])
    first['family_research']['no_progress_cycles'] = 2
    state['response_research'] = first
    current = assess(state, experiments=[experiment('negative_8', 1, 8.)])
    assert current['status'] == 'family_research_needed'
    current['question_id'] = 'response_review'
    chosen = current['operational_fallback']['selected_experiment']
    chosen['family'] = 'linear'
    state['response_research'] = current
    selected = recipe._extend_hypotheses(state, {
        'model_available': False, 'limits': {'max_model_calls': 0}, 'costs': {'model_calls': 0}}, 3, PATH)
    assert selected['response_trial']['prediction'] == 8.
    saved = json.loads(recipe.STATE.read_text())
    assert saved['response_research']['family_research']['selected_test'] == {'crystal': 1, 'frequency': 8.}
    state['response_settings'].append(response(1, 8., False))
    after = assess(state, experiments=[experiment('negative_6', 1, 6.)])
    assert after['family_research']['no_progress_cycles'] == 3


def test_stagnant_live_research_keeps_surviving_families_and_operational_alternative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'shared').mkdir()
    (tmp_path / PATH).write_text(wrapper())
    state = deepcopy(FAILED)
    obligation = {
        'reason': 'completed_tests_changed_members_without_task_or_nonlocal_progress',
        'question': 'Which shared response assumption or missing measurement prevents a new target prediction?',
        'family_statuses': ['SAT_WITNESS', 'UNKNOWN'],
    }
    fallback = {'diagnosis': 'conditional_family_test', 'selected_experiment': {
        'family': 'linear', 'number': 3, 'prediction': 7., 'predictions': {'member': True}}}
    state['response_research'] = {'status': 'family_research_needed',
        'family_research': {'families': [{'id': 'linear', 'status': 'SAT_WITNESS'},
                                       {'id': 'quadratic', 'status': 'UNKNOWN'}],
            'no_progress_cycles': 2, 'research_obligation': obligation},
        'operational_fallback': fallback}
    before = deepcopy(state)

    result = recipe._extend_hypotheses(state, {'model_available': True,
        'limits': {'max_model_calls': 4}, 'costs': {'model_calls': 0},
        'llm_inquiries': {}}, 3, PATH)

    assert result['arguments']['operator'] == 'inquire'
    assert 'response_trial' not in result
    question = result['arguments']['question']['question']
    assert 'shared/reference_extension_3.json' in question
    assert 'shared/reference_recalibration.json' in question
    assert 'Current measured table:' not in question
    assert 'hypotheses failed' not in question
    evidence = json.loads((tmp_path / 'shared/reference_extension_3.json').read_text())
    assert evidence['family_research'] == before['response_research']['family_research']
    assert evidence['operational_fallback'] == fallback
    assert state == before


@pytest.mark.parametrize("limit,spent", [(0, 0), (3, 3)])
def test_no_remaining_calls_explicitly_stops_without_opening_research(tmp_path, monkeypatch, limit, spent):
    monkeypatch.chdir(tmp_path)
    result = recipe._extend_hypotheses(deepcopy(FAILED), {
        "limits": {"max_model_calls": limit}, "costs": {"model_calls": spent},
        "llm_inquiries": {},
    }, 3, PATH)

    assert "live model and remaining calls" in result["stop"]
    assert "action" not in result
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('family_obligation', [None, {
    'reason': 'declared_shared_response_families_refuted',
    'question': 'Which additional measurement distinguishes the contradictory responses?',
    'same_measurement_conflict': [
        {'crystal': 2, 'frequency': 3., 'activated': True, 'evidence': [6]},
        {'crystal': 1, 'frequency': 3., 'activated': False, 'evidence': [10]}],
}])
def test_request_preserves_prior_source_physical_refutation_and_range_failure(tmp_path, monkeypatch, family_obligation):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shared").mkdir()
    source = wrapper()
    (tmp_path / PATH).write_text(source)
    state = deepcopy(FAILED)
    if family_obligation:
        state['response_research'] = {'status': 'family_research_needed',
            'family_research': {'research_obligation': family_obligation}}
    before = deepcopy(state)
    result = recipe._extend_hypotheses(state, {
        "limits": {"max_model_calls": 40}, "costs": {"model_calls": 2},
        "llm_inquiries": {},
    }, 3, PATH)

    assert result["action"] == "run"
    assert result["arguments"]["operator"] == "inquire"
    assert result["arguments"]["max_model_calls"] == 32
    assert result["arguments"]["question"]["id"] == QUESTION
    assert result["arguments"]["question"]["parent_id"] == "task"
    assert (tmp_path / "shared/reference_before_extension_3.py").read_text() == source
    evidence = json.loads((tmp_path / "shared/reference_extension_3.json").read_text())
    assert evidence["physical_tests"] == FAILED["tests"]
    assert evidence["out_of_range_predictions"] == FAILED["out_of_range"]
    assert evidence["rejected_families"] == ["density", "gap"]
    assert evidence["current_program"] == PATH
    assert evidence['family_research'] == (state.get('response_research') or {}).get('family_research')
    assert json.loads((tmp_path / recipe.STATE).read_text()) == before
    assert state == before


def prepared(tmp_path, steps, *, calls=6, readings=(3,)):
    provider = ScriptedProvider(steps)
    agent = DiscoveryWorldAgent(provider, tmp_path, arm="hegel",
                                limits={**LIMITS, "max_model_calls": calls})
    agent.lab = InstrumentLab(readings=readings)
    agent._question({"id": "task", "question": "Use current measurements to tune"})
    agent._execute("query", {"actions": [{"action": "MEASURE"}], "question": {
        "id": "reference_test_3_gap", "question": "Does the old prediction hold?", "parent_id": "task"}},
        agent.lab)
    failed = next(q for q in agent.questions if q["id"] == "reference_test_3_gap")
    agent._record_answer(failed, {"status": "refuted", "answer": "Old prediction contradicted",
                                 "evidence": [0]})
    source = wrapper()
    agent._execute("write", {"path": PATH, "code": source}, agent.lab)
    (tmp_path / recipe.STATE).write_text(json.dumps(FAILED))
    return agent, provider, source


def run_parent(agent):
    return agent._run_procedure(agent.lab, PATH, max_steps=6, query_arguments={
        "procedure_start": 1, "parameters": {"gain": 2}, "research_question": {"id": "task"}})


def test_child_revision_returns_to_actual_entry_and_original_parent_parameters(tmp_path):
    revised = """def select_actions(spec, observations, memory):
    arguments = memory['program_arguments']
    assert arguments['procedure_start'] == 1
    gain = arguments['parameters']['gain']
    reading = observations[0]['feedback']['reading']
    return {'actions': [{'action': 'TUNE', 'arg1': gain * reading}]}
"""
    agent, provider, original = prepared(tmp_path, [
        ("write", {"path": PATH, "code": revised}),
        ("finish", {"lesson": "The revised program can use the current measurement"}),
    ])
    result = run_parent(agent)

    assert result["status"] == "completed" and agent.lab.success
    assert agent.lab.actions == [{"action": "MEASURE"}, {"action": "TUNE", "arg1": 6}]
    assert agent.costs["model_calls"] == len(provider.requests) == 2
    assert all(row["llm_inquiry"]["question_id"] == QUESTION for row in provider.requests)
    assert (tmp_path / PATH).read_text() == revised
    assert (tmp_path / "shared/reference_before_extension_3.py").read_text() == original
    questions = {q["id"]: q for q in agent.questions}
    assert questions["reference_test_3_gap"]["status"] == "refuted"
    assert questions["reference_test_3_gap"]["evidence"] == [0]
    assert questions[QUESTION]["status"] == "open"
    assert questions["task"]["status"] != "supported"
    assert json.loads((tmp_path / recipe.STATE).read_text()) == FAILED


def test_unresolved_child_finish_returns_once_without_research_or_support_loop(tmp_path):
    agent, provider, original = prepared(tmp_path, [
        ("finish", {"lesson": "A new measured variable is still missing"}),
    ])
    result = run_parent(agent)

    assert result["status"] == "method_stopped"
    assert "without a usable extension" in result["steps"][-1]["feedback"]["reason"]
    assert len(provider.requests) == agent.costs["model_calls"] == 1
    assert agent.llm_inquiries[QUESTION]["invocations"] == 1
    assert agent.lab.environment_steps == 1 and not agent.lab.completed
    assert (tmp_path / PATH).read_text() == original
    assert all(q["status"] != "supported" for q in agent.questions)
    # Calling the unresolved parent again must preserve its local record and
    # return to the operator, rather than buying the same unsuccessful loop.
    assert run_parent(agent)["status"] == "method_stopped"
    assert len(provider.requests) == 1


def test_child_uses_only_the_remaining_global_account(tmp_path):
    agent, provider, _ = prepared(tmp_path, [
        ("finish", {"lesson": "Return unresolved"}),
    ], calls=5)
    agent.costs["model_calls"] = 4
    run_parent(agent)

    assert provider.requests[0]["llm_inquiry"]["remaining_model_calls"] == 1
    assert agent.costs["model_calls"] == 5
    assert len(provider.requests) == 1 and not agent.lab.completed


def test_frozen_publication_stops_cleanly_without_provider_despite_positive_call_limit(tmp_path):
    agent, provider, original = prepared(tmp_path, [], calls=6)
    agent.provider = None

    result = run_parent(agent)

    assert result["status"] == "method_stopped"
    assert "live model and remaining calls" in result["steps"][-1]["feedback"]["reason"]
    assert agent.limits["max_model_calls"] == 6 and agent.costs["model_calls"] == 0
    assert not provider.requests and not agent.llm_inquiries
    assert agent.lab.environment_steps == 1 and not agent.lab.completed
    assert (tmp_path / PATH).read_text() == original


FRESH_REFUTATION = """def answer(spec, observations, memory):
    record = observations[-1]
    measured = record['feedback']['reading']
    return {'status': 'supported' if measured == 3 else 'refuted',
            'answer': {'prediction': 3, 'observed': measured}, 'evidence': [record['index']]}
"""


def fresh_refutation_steps():
    return [("query", {"actions": [{"action": "MEASURE"}], "question": {"id": QUESTION}}),
            ("evaluate", {"question_id": QUESTION, "path": "shared/new_hypothesis_test.py",
                          "code": FRESH_REFUTATION})]


def test_fresh_physical_refutation_opens_next_child_then_revised_parent_executes(tmp_path):
    revised = """def select_actions(spec, observations, memory):
    gain = memory['program_arguments']['parameters']['gain']
    return {'actions': [{'action': 'TUNE', 'arg1': gain * observations[0]['feedback']['reading']}]}
"""
    agent, provider, original = prepared(tmp_path, [*fresh_refutation_steps(),
        ("write", {"path": PATH, "code": revised}),
        ("finish", {"lesson": "Revised the contradicted candidate"}),
    ], calls=4, readings=(3, 4))

    result = run_parent(agent)

    assert result["status"] == "completed" and agent.lab.success
    assert [request["llm_inquiry"]["question_id"] for request in provider.requests] == [
        QUESTION, QUESTION, QUESTION + "_2", QUESTION + "_2"]
    assert provider.requests[2]["llm_inquiry"]["remaining_model_calls"] == 2
    assert agent.costs["model_calls"] == len(provider.requests) == 4
    questions = {q["id"]: q for q in agent.questions}
    assert questions[QUESTION]["status"] == "refuted"
    assert questions[QUESTION]["evidence"] == [1]
    assert questions[QUESTION]["opened_before_observation"] == 1
    assert questions[QUESTION + "_2"]["status"] == "open"
    assert questions["reference_test_3_gap"]["status"] == "refuted"
    assert (tmp_path / "shared/reference_before_extension_3.py").read_text() == original
    assert (tmp_path / "shared/reference_before_extension_3_2.py").read_text() == original


def test_refutation_citing_only_preexisting_observation_does_not_restart_research(tmp_path):
    source = """def answer(spec, observations, memory):
    return {'status': 'refuted', 'answer': 'Old observation does not establish this candidate',
            'evidence': [0]}
"""
    agent, provider, _ = prepared(tmp_path, [
        ("evaluate", {"question_id": QUESTION, "path": "shared/old_evidence_test.py", "code": source}),
    ])

    result = run_parent(agent)

    assert result["status"] == "method_stopped"
    assert len(provider.requests) == agent.costs["model_calls"] == 1
    question = next(q for q in agent.questions if q["id"] == QUESTION)
    assert question["status"] == "refuted" and question["evidence"] == [0]
    assert question["opened_before_observation"] == 1
    assert list(agent.llm_inquiries) == [QUESTION]
    assert not agent.lab.completed


def test_new_refuted_candidate_cannot_reset_the_global_account(tmp_path):
    agent, provider, _ = prepared(tmp_path, [*fresh_refutation_steps()], calls=2, readings=(3, 4))

    result = run_parent(agent)

    assert result["status"] == "method_stopped"
    assert len(provider.requests) == agent.costs["model_calls"] == 2
    assert list(agent.llm_inquiries) == [QUESTION]
    assert "live model and remaining calls" in result["steps"][-1]["feedback"]["reason"]
    assert all(q["status"] != "supported" for q in agent.questions)


def test_successive_refutations_preserve_each_prior_source_and_request_next_revision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shared").mkdir()
    (tmp_path / PATH).write_text("# Current researcher revision\n")
    (tmp_path / "shared/reference_before_extension_3.py").write_text("# Original recipe\n")
    (tmp_path / "shared/reference_before_extension_3_2.py").write_text("# First revision\n")
    memory = {
        "limits": {"max_model_calls": 40}, "costs": {"model_calls": 9},
        "llm_inquiries": {qid: {"invocations": 1} for qid in (QUESTION, QUESTION + "_2")},
        "questions": [
            {"id": QUESTION, "status": "refuted", "opened_before_observation": 1, "evidence": [2]},
            {"id": QUESTION + "_2", "status": "refuted", "opened_before_observation": 3,
             "evidence": [4]},
        ],
    }

    result = recipe._extend_hypotheses(deepcopy(FAILED), memory, 3, PATH)

    assert result["arguments"]["question"]["id"] == QUESTION + "_3"
    assert result["arguments"]["max_model_calls"] == 31
    assert (tmp_path / "shared/reference_before_extension_3.py").read_text() == "# Original recipe\n"
    assert (tmp_path / "shared/reference_before_extension_3_2.py").read_text() == "# First revision\n"
    assert (tmp_path / "shared/reference_before_extension_3_3.py").read_text() == "# Current researcher revision\n"
