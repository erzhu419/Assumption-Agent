"""Real retained comparisons supply referenced, criterion-scoped source triggers."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_episodes import extract_episodes, load_source_values
from hegel_machine.cognition.structural_inquiry import _register_native_theories
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from test_question_routing import CONTEXT, ROLES, scene
from test_retained_program_revisions import paired_changes
from test_revision_relations import binary, constant, feature
from test_theory_consolidation import MISSION, agent_scene


def retained_source(path, index, *, threshold=1e-10):
    agent, lab = scene(path)
    agent.config = replace(agent.config, confirmation_mse_threshold=threshold)
    lab.public_spec = lambda: {'controls': ['u'], 'action_kinds': ['trajectory'],
        'max_probe_length': 8, 'feature_bounds': {'u': [-1., 1.]}}
    context = {**CONTEXT, 'source_id': f'actual_retained_source_{index}'}
    question = agent._open_question({'id': 'retained', 'question': 'Compare these executable toy revisions',
        'context_ref': context, 'theory_target': {'role_signature': deepcopy(ROLES),
            'context_ref': context, 'context_family_ref': 'retained_fixture', 'output_unit': 'V'}})
    before = (binary('add', Program('abs', children=(feature(),)), constant(2)) if index == 0
              else binary('sub', feature(), constant(3)))
    after = binary('add', before, constant(-1 if index == 0 else 2))
    models = ['native/old.py', 'native/new.py']
    question['native_condition_research'] = []
    for round_index, (program, model) in enumerate(zip((before, after), models)):
        agent._save(model, export_temporal_program(program, feature_names=['u']))
        question['native_condition_research'].append({'round_index': round_index, 'status': 'candidate',
            'fit_record_indices': [0], 'fit_evidence': [0, 1], 'observed_record_count': 1,
            'exported_models': [], 'compute_cost': 0, 'proposed_experiments': [], 'evidence': [],
            'search_stages': [], 'candidates': [{'program': program.to_dict(), 'model': model}]})
        _register_native_theories(agent, question, agent.questions.index(question), round_index)
    agent._execute('evaluate', {'model': models[0]}, lab)
    agent._save('shared/compare.py', 'def select_actions(spec, observations, memory):\n    return ' + repr({
        'action': 'run', 'arguments': {'operator': 'research_choice', 'models': models,
                                     'research_question': {'id': question['id']}}}) + '\n')
    execute_procedure(agent, lab, 'shared/compare.py', max_steps=1, record_events=True)
    change, = paired_changes(extract_episodes(agent))
    agent._prepare_program_state()
    (path / 'training.json').write_text(json.dumps(agent.result('method_stopped')))
    identifier = next(key for key, value in agent.theory_index.items()
                      if value['source_ref'].get('round_index') == 1)
    return agent, lab, change, {'file': str(path / 'training.json'), 'json_path': [],
                              'source_theory_ids': [identifier]}, f'{path / "research_state.json"}::{identifier}'


def build(target, lab, references, **extra):
    return target._execute('run', {'operator': 'theory_consolidation', 'object_type': 'program_revision_relation',
        'theme': 'actual_retained_edits', 'common_source_refs': references, 'mission': deepcopy(MISSION),
        'research_question': {'id': 'trigger_sources', 'question': 'Construct the retained edit and its cited trigger'},
        'allocation': {'compute': 10000, 'max_groups': 8}, **extra}, lab)


def test_common_retention_refs_produce_explicit_mse_witnesses_and_cache_without_new_predictions(tmp_path, monkeypatch):
    sources = [retained_source(tmp_path / f'source_{index}', index) for index in range(2)]
    for agent, _, change, _, _ in sources:
        assert set(change['source_comparison']) == {'before_evaluation_ref', 'after_evaluation_ref', 'criterion_ref'}
        assert '"observation"' not in json.dumps(change)
        assert agent.events[0]['feedback']['evaluations'][1]['retention_decision']['status'] == 'candidate_retained'
    # The saved training outcome still declares the original criterion even
    # when the separate state no longer includes that limit.
    state_path = sources[0][0].workspace / 'research_state.json'
    state = json.loads(state_path.read_text())
    state['limits'].pop('confirmation_mse_threshold')
    state_path.write_text(json.dumps(state))
    target, lab = agent_scene(tmp_path / 'target')
    calls = []
    from hegel_machine.cognition import revision_relations
    original = revision_relations._discrepancy

    def counted(program, example, budget):
        result = original(program, example, budget)
        calls.append(result['status'])
        return result

    monkeypatch.setattr(revision_relations, '_discrepancy', counted)
    agent, _, _, _, identifier = sources[0]
    explicit = {'source_ref': {'file': str(agent.workspace / 'laboratory_observations.json'),
                              'json_path': [0, 'rows', 0]},
                'observation': deepcopy(agent.observations[0]['rows'][0]), 'tolerance': .25}
    arguments = {'counterexamples': {identifier: [explicit]}}
    before_costs = deepcopy(target.costs)
    result = build(target, lab, [item[3] for item in sources], **arguments)
    assert result['status'] == 'completed' and len(calls) == 6
    candidate, = target.theory_consolidation['candidates'].values()
    checks = candidate['trigger_checks']
    derived = [check for check in checks if 'criterion' in check]
    assert len(derived) == 2 and any(check['tolerance'] == .25 for check in checks)
    for check in derived:
        criterion = check['criterion']
        assert criterion['kind'] == 'paid_mse_single_residual_witness'
        assert criterion['evaluated_rows'] == 2 and criterion['threshold'] == 1e-10
        assert criterion['criterion_ref']['file'].endswith('training.json')
        assert check['tolerance'] ** 2 == pytest.approx(2e-10)
        assert check['before']['failed'] and check['after']['failed']
        assert check['before']['prediction'] != check['after']['prediction']
        assert check['before']['squared_error'] > 2e-10 and check['after']['squared_error'] > 2e-10
        retained = load_source_values([criterion['criterion_ref'], criterion['before_evaluation_ref'],
            criterion['before_residual_ref'], criterion['paid_row_ref']], target.workspace)
        assert retained[0]['value'] == 1e-10
        assert retained[1]['value']['mse'] > 1e-10
        assert retained[2]['value']['squared_error'] > 2e-10
        assert retained[3]['value']['target'] == check['before']['target']
    assert '"observation"' not in json.dumps(candidate)
    assert not candidate['active_library'] and not candidate['scientific_support_added']
    assert target.costs['compute_cost'] - before_costs['compute_cost'] == result['new_compute_cost'] >= len(calls)
    spent, library = deepcopy(target.costs), (target.workspace / 'shared/theory_library.json').read_text()
    repeated = build(target, lab, [item[3] for item in sources], **arguments)
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0 and len(calls) == 6
    assert target.costs == spent and (target.workspace / 'shared/theory_library.json').read_text() == library
    assert lab.paid == 0 and target.costs['model_calls'] == 0 and not target.observations
    assert all(source_lab.paid == 2 and source_agent.costs['model_calls'] == 0
               for source_agent, source_lab, *_ in sources)


@pytest.mark.parametrize('missing', ['criterion', 'sufficient_single_row'])
def test_missing_source_trigger_contract_keeps_construction_and_aggregate_refs_but_no_default_check(tmp_path, missing):
    sources = [retained_source(tmp_path / f'source_{index}', index,
               threshold=3. if index == 0 else 6.) for index in range(2)]
    if missing == 'criterion':
        for agent, *_ in sources:
            for name in ('research_state.json', 'training.json'):
                path = agent.workspace / name
                state = json.loads(path.read_text())
                state['limits'].pop('confirmation_mse_threshold')
                path.write_text(json.dumps(state))
    target, lab = agent_scene(tmp_path / 'target')
    result = build(target, lab, [item[3] for item in sources])
    assert result['status'] == 'author_work_request'
    candidate, = target.theory_consolidation['candidates'].values()
    assert candidate['trigger_checks'] == [] and not candidate['active_library']
    reason = ('original_paid_mse_criterion_required' if missing == 'criterion'
              else 'saved_single_residual_paid_mse_failure_witness_unavailable')
    assert any(item['reason'] == reason for item in result['author_work'])
    if missing == 'sufficient_single_row':
        for source in candidate['source_outcomes']:
            comparison = source['source_comparison']
            before, threshold = [item['value'] for item in load_source_values(
                [comparison['before_evaluation_ref'], comparison['criterion_ref']], target.workspace)]
            assert before['mse'] > threshold
    assert lab.paid == 0 and not target.observations and target.costs['model_calls'] == 0


def test_live_local_revision_uses_current_events_limits_and_paid_rows_without_prediction(tmp_path, monkeypatch):
    from hegel_machine.cognition import research_episodes
    from hegel_machine.cognition.question_routing import resolved_theories
    from hegel_machine.cognition.theory_consolidation import _revision_entries

    agent, lab, _, _, _ = retained_source(tmp_path, 0)
    # The previous request contains the real earlier event prefix, while the
    # just-executed procedure's retention event is already in live memory.
    prior_request = {'events': deepcopy(agent.events[:-1])}
    agent._save('research_context.json', json.dumps(prior_request))
    assert prior_request['events'] == [] and len(agent.events) == 1
    costs, observations = deepcopy(agent.costs), deepcopy(agent.observations)

    def no_prediction(*args, **kwargs):
        raise AssertionError('source trigger extraction must not rerun a Program')

    def no_disk_sources(references, workspace):
        assert references == [], 'the supported local source refs must resolve from live memory'
        return []

    monkeypatch.setattr(Program, 'predict', no_prediction)
    monkeypatch.setattr(research_episodes, 'load_source_values', no_disk_sources)
    intake = extract_episodes(agent)
    identifier = next(key for key, value in agent.theory_index.items()
                      if value['source_ref'].get('round_index') == 1)
    episode = next(item for item in intake['episodes'] if identifier in item['theory_ids'])
    pool = {item['id']: item for item in resolved_theories(agent)}
    entry, = _revision_entries(agent, [{**episode, 'theory_id': identifier}], pool, retained=[])
    witness, = entry['counterexamples']
    assert witness['criterion']['kind'] == 'paid_mse_single_residual_witness'
    assert witness['criterion']['threshold'] == agent.config.confirmation_mse_threshold == 1e-10
    assert witness['criterion']['evaluated_rows'] == 2
    assert witness['criterion']['before_evaluation_ref'] == {
        'file': 'research_context.json', 'json_path': ['events', 0, 'feedback', 'evaluations', 0]}
    assert witness['criterion']['criterion_ref'] == {
        'file': 'research_state.json', 'json_path': ['limits', 'confirmation_mse_threshold']}
    assert witness['source_ref'] == {
        'file': 'laboratory_observations.json', 'json_path': [0, 'rows', 0]}
    assert witness['observation'] == observations[0]['rows'][0]
    assert witness['tolerance'] ** 2 == pytest.approx(2e-10)
    assert agent.costs == costs and agent.observations == observations and lab.paid == 2
