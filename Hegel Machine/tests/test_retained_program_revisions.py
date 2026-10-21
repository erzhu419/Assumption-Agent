"""Actual toy Common retention decisions supply references, not generated ancestry."""

from copy import deepcopy
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_episodes import extract_episodes, load_common_sources, load_source_values
from hegel_machine.cognition.structural_inquiry import _register_native_theories
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from test_question_routing import CONTEXT, ROLES, scene


def retained_scene(path, *, old_native=True):
    agent, lab = scene(path)
    lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
                              'max_probe_length': 8, 'feature_bounds': {'u': [-1., 1.]}}
    context = {**CONTEXT, 'source_id': 'retention-toy-source'}
    question = agent._open_question({'id': 'retained', 'question': 'Compare supplied toy programs',
        'context_ref': context, 'theory_target': {'role_signature': deepcopy(ROLES),
            'context_ref': context, 'output_unit': 'V'}})
    programs = [Program('const', value=0.), Program('feature', name='u')]
    models = ['native/round_0/old.py' if old_native else 'legacy.py', 'native/round_1/new.py']
    question['native_condition_research'] = []
    for index, (program, model) in enumerate(zip(programs, models)):
        agent._save(model, export_temporal_program(program, feature_names=['u']))
        question['native_condition_research'].append({'round_index': index,
            'status': 'candidate', 'fit_record_indices': [0], 'fit_evidence': [0, 1],
            'observed_record_count': 1, 'exported_models': [], 'compute_cost': 0,
            'proposed_experiments': [], 'evidence': [], 'search_stages': [],
            'candidates': [{'program': program.to_dict(), 'model': model}] if index or old_native else []})
        _register_native_theories(agent, question, agent.questions.index(question), index)
    agent._execute('evaluate', {'model': models[0]}, lab)
    agent._save('shared/compare.py', "def select_actions(spec, observations, memory):\n    return " + repr({
        'action': 'run', 'arguments': {'operator': 'research_choice', 'models': models,
                                     'research_question': {'id': question['id']}}}) + '\n')
    execute_procedure(agent, lab, 'shared/compare.py', max_steps=1, record_events=True)
    return agent, lab


def paired_changes(intake):
    return [change for episode in intake['episodes'] for change in episode['actual_changes']
            if change.get('before_ref')]


def test_actual_retained_native_replacement_exports_exact_source_refs_and_paid_scope(tmp_path):
    agent, lab = retained_scene(tmp_path)
    before_costs = deepcopy(agent.costs)
    intake = extract_episodes(agent)
    change, = paired_changes(intake)
    assert change['before_ref']['json_path'] == [
        'questions', 1, 'native_condition_research', 0, 'candidates', 0, 'program']
    assert change['source_ref']['json_path'] == [
        'questions', 1, 'native_condition_research', 1, 'candidates', 0, 'program']
    assert change['action_ref'] == {'file': 'research_context.json', 'json_path': [
        'events', 0, 'feedback', 'evaluations', 1, 'retention_decision']}
    assert change['evidence_refs'] == [{'file': 'laboratory_observations.json', 'json_path': [0]}]
    assert change['scope']['same_paid_rows'] and change['scope']['evaluated_rows'] == 2
    assert change['scope']['independent_confirmation'] is False
    assert '"op"' not in json.dumps(change)
    # The real shared-source entry anchors both Programs and the actual decision.
    agent._prepare_program_state()
    result = agent.result('method_stopped')
    (tmp_path / 'training.json').write_text(json.dumps(result))
    loaded = load_common_sources([{'file': str(tmp_path / 'training.json'), 'json_path': []}], tmp_path)
    imported, = paired_changes(loaded)
    values = load_source_values([imported['before_ref'], imported['source_ref'], imported['action_ref']], tmp_path)
    assert [value['value'] for value in values[:2]] == [
        Program('const', value=0.).to_dict(), Program('feature', name='u').to_dict()]
    assert values[2]['value']['status'] == 'candidate_retained'
    assert agent.costs == before_costs and lab.paid == 2
    assert not agent.confirmations and agent.costs['model_calls'] == 0


def test_cached_summary_and_actual_incumbent_retention_do_not_repeat_revision(tmp_path):
    agent, lab = retained_scene(tmp_path)
    assert len(paired_changes(extract_episodes(agent))) == 1
    execute_procedure(agent, lab, 'shared/compare.py', max_steps=1, record_events=True)
    assert agent.events[-1]['feedback']['evaluations'][1]['retention_decision']['status'] == 'incumbent_retained'
    assert agent.events[-1]['feedback']['evaluated_model']['selection']['status'] == 'candidate_retained'
    assert len(paired_changes(extract_episodes(agent))) == 1
    # Retained bank additions keep their original fact without invented parents.
    additions = [change for episode in extract_episodes(agent)['episodes']
                 for change in episode['actual_changes'] if change['kind'] == 'program_changed']
    assert additions and all('before_ref' not in change for change in additions)


def test_first_python_to_native_retention_has_no_invented_before_program(tmp_path):
    agent, _ = retained_scene(tmp_path, old_native=False)
    decision = agent.events[0]['feedback']['evaluations'][1]['retention_decision']
    assert decision['status'] == 'candidate_retained'
    assert decision['incumbent']['source_model'] == 'legacy.py'
    assert paired_changes(extract_episodes(agent)) == []
