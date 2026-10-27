"""Native representation research consumes only its declared training split."""

from dataclasses import replace
from pathlib import Path
import json
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, TaskSpec
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.representation_inquiry import (
    execute_native_representation_inquiry, load_native_representation_result,
    load_native_interrupted_fit,
)
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.representation_diagnostics import (
    representation_ambiguity, representation_evidence_state,
)


def agent_with_records(tmp_path, *, trajectories=5):
    agent = TemporalOperatorAgent(None, tmp_path, arm='hegel', config=replace(
        InquiryLLMConfig(), compute_budget=150000, confirmation_mse_threshold=1e-10))
    rng = Random(71)
    agent.observations = []
    agent.rows = []
    for index in range(trajectories):
        history, rows = [], []
        for _ in range(12):
            value = rng.uniform(-1, 1)
            row = {'features': {'u': value}, 'history': list(history),
                   'target': value + sum(point['u'] for point in history[-2:])}
            rows.append(row)
            history.append({'u': value})
        agent.observations.append({'action': {'kind': 'trajectory',
            'sequence': [row['features'] for row in rows]}, 'rows': rows,
            'row_indices': list(range(len(agent.rows), len(agent.rows) + len(rows)))})
        agent.rows.extend(rows)
    return agent


def measured_failure_basis(agent, distinction):
    refs = []
    for index in (3, 4):
        identifier = f'failed_forecast:{index}'
        agent.questions.append({'id': identifier, 'kind': 'evidence', 'meaning_version': 1,
            'query_semantics': {'query_expression': {'op': 'AllOnFiniteSet'}},
            'semantic_answer': {'version': 1, 'finite_status': 'observed_false',
                'value': False, 'evidence': agent.observations[index]['row_indices']}})
        refs.append({'kind': 'question', 'id': identifier, 'version': 1, 'answer_version': 1})
    return {'failure_refs': refs, 'distinction_program': distinction.to_dict()}


def competing_representation_basis(agent, tmp_path, operation='window_sum'):
    from hegel_machine.cognition.models import Observation
    drive = Program('feature', name='u')
    alternative = (Program('accumulate', value=.5, children=(drive, Program('const', value=0.)))
                   if operation == 'accumulate' else Program(operation, lag=3, children=(drive,)))
    rows = [Observation(row['features'], row['target'], tuple(row['history']))
            for row in agent.observations[4]['rows']]
    selected = {'program': alternative.to_dict(), 'best_retained_id': 'old',
        'validation_predictions': [alternative.predict(row) for row in rows],
        'best_retained_validation_predictions': [row.features['u'] for row in rows]}
    path = tmp_path / 'paid_alternative.json'
    path.write_text(json.dumps({'selected': selected}))
    return {'alternative_ref': {'file': str(path), 'json_path': ['selected']}}


def learned_representation_basis(tmp_path, *, parameter_only=False):
    from hegel_machine.cognition.research_programs import consolidate_research_programs
    from test_question_routing import ROLES, CONTEXT
    entries = []
    for index in range(2):
        directory = tmp_path / f'learned_source_{index}'
        directory.mkdir()
        term = (None if parameter_only else Program('window_sum', lag=3,
            children=(Program('feature', name='u'),)))
        _, reference = retained_procedure(directory, intercept=float(index), term=term)
        saved = json.loads(Path(reference['file']).read_text())['candidates'][0]
        entries.append({**saved, 'source_ref': reference, 'source_id': f'source:{index}',
            'episode_id': f'episode:{index}', 'context_ref': CONTEXT,
            'role_signature': ROLES, 'output_unit': 'V'})
    method, = consolidate_research_programs(entries, allocation={'compute': 1000})['candidates']
    path = tmp_path / 'learned_method.json'
    path.write_text(json.dumps({'candidate': method}))
    return {'learned_method_ref': {'file': str(path), 'json_path': ['candidate']}}


def run(agent, question, **kwargs):
    return execute_native_representation_inquiry(agent, question,
        spec=TaskSpec('development', 'public response', ('u',), (), ()),
        parent_programs=kwargs.pop('parent_programs', {'old': Program('feature', name='u')}),
        validation_records=kwargs.pop('validation_records', [4]),
        compute_limit=kwargs.pop('compute_limit', 100000), max_candidates=4, **kwargs)


def test_native_search_selects_executable_temporal_draft_and_changes_actual_predictions(tmp_path):
    agent = agent_with_records(tmp_path)
    question = {'id': 'representation:old', 'kind': 'representation'}
    result = run(agent, question)
    assert result['status'] == 'native_representation_draft_selected'
    assert result['origin'] == 'not_exposed'
    assert result['selected']['validation_mse'] < 1e-10
    assert result['selected']['best_retained_validation_mse'] > 1e-2
    assert result['selected']['changed_prediction_rows'] > 0
    assert result['selected']['max_prediction_change'] > 0
    assert result['comparison_completed']
    assert result['provider_calls'] == result['new_observations'] == 0
    assert result['compute_cost'] == agent.costs['compute_cost'] <= 100000
    assert result['native_host_seconds_before_persistence'] > 0
    assert agent.costs['python_execution_seconds'] == 0
    receipt = question['native_representation_research'][-1]
    assert load_native_representation_result(receipt['source_ref'], agent.workspace) == result
    canonical = json.loads((agent.workspace / receipt['source_ref']['file']).read_text())
    assert 'program' not in canonical['selected']
    assert canonical['selected']['program_ref']['json_path'] == [
        'candidates', int(result['selected']['id'].rsplit(':', 1)[1]), 'program']
    assert receipt['candidate_count'] == len(result['candidates'])
    assert all('program' not in candidate and 'research_program' not in candidate
               for candidate in receipt['candidates'])
    assert 'program' not in receipt['selected'] and 'research_program' not in receipt['selected']
    assert question['native_representation_split']['training_records'] == [0, 1, 2, 3]
    selection = result['selected']['selection']
    assert selection['status'] == 'working_draft' and not selection['independent_confirmation']
    assert selection['fitting_records'] == [0, 1, 2, 3]
    assert selection['selection_records'] == [4]
    assert selection['comparison_scope'] == 'declared_parents_and_current_native_candidates'
    assert selection['uncompared_archive_validation'] == 'not_established'
    assert 'status' not in question  # Installation and child resolution belong to the caller.


def test_predeclared_utility_scope_changes_executable_candidate_without_fitting_on_validation(tmp_path, monkeypatch):
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.models import Observation

    agent = agent_with_records(tmp_path)
    for index, record in enumerate(agent.observations):
        record['rows'] = [{'features': {'u': value + index * .01}, 'history': [],
            'target': max(0., value + index * .01)} for value in (-1., .5, 1., 1.5)]
        record['action']['sequence'] = [row['features'] for row in record['rows']]
    agent.rows = [row for record in agent.observations for row in record['rows']]

    def compose(self, spec, rows, budget, parents, actions):
        program = Program('feature', name='u')
        return [{'candidate': Candidate('candidate_u', program, 'control'),
                 'training_loss': loss(program, rows, 0., .5)}]

    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', compose)
    question = {'id': 'scope', 'kind': 'program_revision', 'native_representation_plan': {
        'scope_plan': {'purpose': 'relative_utility', 'objective': 'pointwise_improvement',
                       'max_compute': 5000, 'max_history': 4}}}
    result = run(agent, question, parent_programs={'old': Program('const', value=0.)})
    scope = result['relative_utility_research']
    assert scope['fitting_records'] == [0, 1, 2, 3] and scope['selection_records'] == [4]
    assert scope['scope_purpose'] == 'relative_utility'
    assert scope['application_scope_status'] == 'not_assessed'
    assert 0 < scope['compute_cost'] <= scope['compute_limit'] == 5000
    assert result['selected']['id'] == scope['candidate_id']
    assert result['selected']['validation_mse'] < 1e-10
    scoped = Program.from_dict(result['selected']['program'])
    assert scoped.op == 'if'
    assert scoped.predict(Observation({'u': -.5}, 500., ())) == 0.
    assert result['selected']['selection']['inherited_parameter_provenance'] == 'original_component_fitting_views_not_established'
    assert result['compute_cost'] == agent.costs['compute_cost'] <= 100000

    # Held-out targets may change adoption, never which training-only scope
    # candidate is constructed or what its selector reads.
    for row in agent.observations[4]['rows']:
        row['target'] = -10.
    changed = run(agent, question, parent_programs={'old': Program('const', value=0.)})
    assert changed['relative_utility_research']['selector'] == scope['selector']
    assert changed['selected']['retained']
    scoped_candidate = next(item for item in changed['candidates'] if item.get('relative_utility_scope'))
    assert scoped_candidate['program'] == scoped.to_dict()

    # An explicitly declared mean-loss selector is constant here. Its branch
    # already has a paid score; no semantically identical if-program is added.
    question['native_representation_plan']['scope_plan']['objective'] = 'mean_loss'
    mean_scope = run(agent, question, parent_programs={'old': Program('const', value=0.)})
    utility = mean_scope['relative_utility_research']
    assert utility['selector']['op'] == 'const' and utility['reused_candidate_ref']
    assert len(mean_scope['candidates']) == 1 and 'candidate_id' not in utility


def test_unfunded_scope_retains_comparison_and_does_not_increase_budget(tmp_path):
    agent = agent_with_records(tmp_path)
    result = run(agent, {'id': 'scope', 'kind': 'program_revision', 'native_representation_plan': {
        'scope_plan': {'purpose': 'relative_utility', 'objective': 'mean_loss', 'max_compute': 1}}})
    assert result['comparison_completed']
    scope = result['relative_utility_research']
    assert scope['status'] == 'insufficient_budget' and 'candidate_id' not in scope
    assert scope['compute_cost'] == 0 and scope['compute_limit'] == 1
    assert result['compute_cost'] <= 100000


def test_split_is_persisted_before_proposal_and_validation_targets_do_not_enter_search(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    question = {'id': 'representation:old'}
    held_ids = {id(row) for row in agent.observations[4]['rows']}
    train_targets = [row['target'] for record in agent.observations[:4] for row in record['rows']]
    calls = []

    def coordinates(self, spec, rows, budget, actions):
        assert question['native_representation_split']['validation_records'] == [4]
        assert [row.target for row in rows] == train_targets
        assert all(id(row) not in held_ids for row in rows)
        calls.append('coordinates')
        return []

    def compose(self, spec, rows, budget, parents, actions):
        assert [row.target for row in rows] == train_targets
        calls.append('composition')
        return []

    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', coordinates)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', compose)
    result = run(agent, question)
    assert calls == ['coordinates', 'composition']
    assert result['status'] == 'native_representation_parent_retained'


def test_every_retained_parent_competes_and_comparison_is_reserved_before_search(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    question = {'id': 'representation:old'}
    parents = {'weak': Program('const', value=100.), 'strong': Program('feature', name='u')}

    def exhausting(self, spec, rows, budget, actions):
        assert budget.spend(budget.remaining)
        return [{'candidate': Candidate('worse', Program('const', value=200.), 'test'),
                 'training_loss': 50000.}]

    def no_more(self, spec, rows, budget, parents, actions):
        assert budget.spend(budget.remaining)
        return []

    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', exhausting)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', no_more)
    result = run(agent, question, parent_programs=parents, compute_limit=2000)
    assert {entry['id'] for entry in result['retained_comparison']} == {'weak', 'strong'}
    assert len(result['candidates']) == 1
    assert result['selected']['id'] == 'strong'
    assert result['comparison_completed']
    assert result['compute_cost'] <= 2000


def test_insufficient_budget_does_not_fit_or_select_partial_pool(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)

    def never(*args):
        raise AssertionError('unfunded search ran')

    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', never)
    result = run(agent, {'id': 'representation:old'}, compute_limit=1)
    assert result['status'] == 'native_representation_inconclusive_budget'
    assert result['selected'] is None
    assert result['compute_cost'] == 0


def test_equal_public_reset_records_stay_together_regardless_of_targets(tmp_path):
    agent = agent_with_records(tmp_path)
    duplicate = {'action': agent.observations[0]['action'],
        'rows': [{**row, 'target': row['target'] + .1} for row in agent.observations[0]['rows']],
        'row_indices': list(range(len(agent.rows), len(agent.rows) + 12))}
    agent.observations.append(duplicate)
    agent.rows.extend(duplicate['rows'])
    result = run(agent, {'id': 'representation:old'}, validation_records=[5], compute_limit=1)
    assert result['split']['validation_records'] == [0, 5]
    assert result['split']['training_records'] == [1, 2, 3, 4]


def test_failed_native_direction_retains_paid_work_and_failure(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    question = {'id': 'representation:old'}

    def broken(self, spec, rows, budget, actions):
        budget.spend(17)
        raise ValueError('actual fitting error')

    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', broken)
    with pytest.raises(ValueError, match='actual fitting error'):
        run(agent, question)
    receipt = question['native_representation_research'][-1]
    assert receipt['status'] == 'native_representation_execution_failed'
    assert receipt['compute_cost'] == 60 + 17
    assert receipt['compute_cost'] == agent.costs['compute_cost']
    assert receipt['selected'] is None
    restored = load_native_representation_result(receipt['source_ref'], agent.workspace)
    assert restored['error'] == 'ValueError: actual fitting error'
    assert restored['compute_cost'] == receipt['compute_cost']


def test_program_revision_uses_actual_parent_only_and_keeps_counterexample_origin(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    actual = Program('feature', name='u')
    origin = {'kind': 'frozen_prediction_counterexample', 'row_indices': [48, 49]}
    question = {'id': 'revision:actual_new_parent', 'kind': 'program_revision', 'origin': origin}
    calls = []
    original = UnifiedResearchAgent._program_claims

    def coordinates(*args):
        raise AssertionError('a narrow revision acquired an undeclared search direction')

    def compose(self, spec, rows, budget, parents, actions):
        assert parents == [actual]
        assert len(rows) == 48
        assert question['native_representation_split']['validation_records'] == [4]
        calls.append('retained_composition')
        return original(self, spec, rows, budget, parents, actions)

    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', coordinates)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', compose)
    result = run(agent, question, parent_programs={'actual_new_parent': actual})
    assert calls == ['retained_composition']
    assert result['research_kind'] == 'program_revision'
    assert result['origin'] == origin
    assert result['parent_theory_ids'] == ['actual_new_parent']
    assert [entry['direction'] for entry in result['search_directions']] == ['retained_composition']
    assert result['comparison_completed'] and result['candidates']


@pytest.mark.parametrize(('diagnostic', 'state'), [
    ({'status': 'not_exposed'}, 'not_exposed'),
    ({'status': 'conflicting_targets', 'conflicting_groups': 1}, 'proven_insufficient_on_evidence'),
    ({'status': 'no_exact_conflicts'}, 'not_refuted_on_current_evidence'),
    ({'status': 'no_exact_conflicts', 'evidence_state': 'supported_for_declared_tests',
      'declared_tests': [{'source_ref': {'event_index': 1}, 'passed': True}]},
     'supported_for_declared_tests'),
    ({'status': 'no_exact_conflicts', 'evidence_state': 'supported_for_declared_tests'},
     'not_refuted_on_current_evidence'),
    ({'status': 'no_exact_conflicts', 'evidence_state': 'supported_for_declared_tests',
      'declared_tests': [{'source_ref': {'event_index': 1}, 'passed': False}]},
     'not_refuted_on_current_evidence'),
])
def test_representation_support_requires_more_than_absent_collisions(diagnostic, state):
    assert representation_evidence_state(diagnostic) == state


def test_unique_inputs_with_public_missing_distinction_open_finite_alternative(tmp_path):
    agent = agent_with_records(tmp_path)
    diagnostic = representation_ambiguity([row['features']['u'] for row in agent.rows], agent.rows)
    assert diagnostic['conflicting_groups'] == 0
    question = {'id': 'revision:missing_distinction', 'kind': 'program_revision',
        'representation_diagnostic': diagnostic,
        'use_sites': [{'kind': 'supply_component', 'model': 'parent.py', 'slot': 'representation'}],
        'representation_research_basis': measured_failure_basis(
            agent, Program('lag', name='u', lag=1))}
    result = run(agent, question)
    assert result['representation_evidence_state'] == 'not_refuted_on_current_evidence'
    assert result['alternative_representation_basis'] == (
        'prospective_failures_with_public_unrepresented_distinction')
    assert [entry['direction'] for entry in result['search_directions']] == [
        'coordinates', 'retained_composition']
    assert result['selected']['validation_mse'] < 1e-10
    assert result['selected']['changed_prediction_rows'] > 0
    assert result['provider_calls'] == result['new_observations'] == 0


def test_parameter_only_revision_does_not_force_history_search(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    for row in agent.rows:
        row['target'] = 3 * row['features']['u'] + .2

    def unexpected(*args):
        raise AssertionError('parameter-only failure forced a representation expansion')

    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', unexpected)
    question = {'id': 'revision:coefficients', 'kind': 'program_revision',
        'representation_diagnostic': representation_ambiguity(
            [row['features']['u'] for row in agent.rows], agent.rows),
        'representation_research_basis': measured_failure_basis(
            agent, Program('feature', name='u'))}
    result = run(agent, question)
    assert result['alternative_representation_basis'] is None
    assert [entry['direction'] for entry in result['search_directions']] == ['retained_composition']
    # The behavior under test is the optional representation decision; the
    # existing finite composition prefix need not solve every coefficient fit.
    assert result['comparison_completed']


def test_native_interruption_restores_paid_fit_without_refitting_and_marks_unfinished_column(tmp_path, monkeypatch):
    from copy import deepcopy
    from hegel_machine.cognition.research_programs import _Fitter
    agent = agent_with_records(tmp_path)
    question = {'id': 'revision:interrupted', 'kind': 'program_revision'}
    parent = Program('feature', name='u')
    missing = Program('lag', name='u', lag=99)
    pending = Program('mul', children=(parent, parent))
    expected = {}
    original_predict = __import__('hegel_machine.cognition.research_programs',
        fromlist=['predict_rows']).predict_rows

    def predict(program, rows):
        if program == pending:
            raise RuntimeError('actual column interruption')
        return original_predict(program, rows)

    def interrupted(self, spec, rows, budget, parents, actions):
        fitter = _Fitter(rows, budget, .5)
        assert fitter.fit([missing]) is None
        fitted = fitter.fit(parents)
        expected.update(program=fitted['program'].to_dict(),
            coefficients=list(fitted['coefficients']), loss=fitted['loss'],
            columns=deepcopy(fitter.columns[parent]))
        fitter.column(pending)

    monkeypatch.setattr('hegel_machine.cognition.research_programs.predict_rows', predict)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', interrupted)
    with pytest.raises(RuntimeError, match='actual column interruption'):
        run(agent, question, parent_programs={'actual': parent})
    receipt = question['native_representation_research'][-1]
    assert 'interrupted_work' not in receipt
    complete = load_native_representation_result(receipt['source_ref'], agent.workspace)
    work = complete['interrupted_work']
    fitter, = work['fitters']
    assert fitter['rows'] == 48 and fitter['budget']['completed_fits'] == 1
    assert fitter['budget']['spent'] == 5 * 48
    assert receipt['compute_cost'] == 60 + 5 * 48
    assert receipt['search_directions'][0]['completed_fits'] == 1
    assert fitter['unfinished_columns'] and fitter['failed_fit_terms']
    failed = [entry for entry in fitter['columns'] if entry['values'] is None]
    assert len(failed) == 1 and failed[0]['status'].startswith('failed_')
    assert all(entry['term'] != fitter['unfinished_columns'][0]['term']
               for entry in fitter['columns'])
    assert work['frames'][-2]['function'] == 'column'

    def no_computation(*args, **kwargs):
        raise AssertionError('paid fit recovery recomputed a fit or prediction')

    monkeypatch.setattr('hegel_machine.cognition.research_programs._linear_fit', no_computation)
    monkeypatch.setattr('hegel_machine.cognition.research_programs.predict_rows', no_computation)
    restored = load_native_interrupted_fit(receipt['interrupted_work_ref'], agent.workspace)
    assert restored['program'] == expected['program']
    assert restored['coefficients'] == expected['coefficients']
    assert restored['loss'] == expected['loss'] and restored['rows'] == 48
    assert agent.costs['compute_cost'] == receipt['compute_cost']


def retained_procedure(tmp_path, *, intercept=2., slope=3., term=None):
    """A saved actual-parent graph, with no fitting in the test fixture."""
    from hegel_machine.cognition.continuous_state import _sum_terms
    from hegel_machine.cognition.research_programs import _entry
    term = term or Program('feature', name='u')
    parent = _sum_terms([term], [slope], intercept)
    fit = {'fit_terms': [term], 'terms': [term], 'program': parent,
           'coefficients': [intercept, slope], 'loss': .5}
    claim = _entry(fit, [], 48, 144)
    path = tmp_path / 'saved_parent_procedure.json'
    path.write_text(json.dumps({'candidates': [{'program': parent.to_dict(),
        'research_program': claim['research_program']}]}))
    return parent, {'file': str(path), 'json_path': ['candidates', 0]}


def test_actual_parent_procedure_refits_training_once_and_keeps_source_graph(tmp_path, monkeypatch):
    from hegel_machine.cognition.research_programs import replay_research_program
    agent = agent_with_records(tmp_path)
    parent, reference = retained_procedure(tmp_path)
    question = {'id': 'revision:actual', 'kind': 'program_revision'}
    calls = []
    train_targets = [row['target'] for record in agent.observations[:4] for row in record['rows']]

    def replay(payload, rows, budget, **kwargs):
        assert [row.target for row in rows] == train_targets
        calls.append(len(rows))
        return replay_research_program(payload, rows, budget, **kwargs)

    def compose(self, spec, rows, budget, parents, actions, *, replayed):
        assert parents == [parent] and len(replayed) == 1
        return replayed

    monkeypatch.setattr('hegel_machine.cognition.research_programs.replay_research_program', replay)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', compose)
    result = run(agent, question, parent_programs={'actual': parent},
                 retained_research_program_refs=[reference, reference])
    assert calls == [48]
    actions = [action for action in result['actions']
               if action['action'] == 'replay_retained_parent_research_program']
    assert [action['status'] for action in actions] == [
        'refitted_requires_validation', 'duplicate_procedure_reused']
    assert [action['compute_cost'] for action in actions] == [144, 0]
    assert actions[0]['validation_rows_used_for_fit'] == 0
    assert actions[1]['reused_source_ref'] == reference
    assert result['compute_cost'] == 60 + 144 + 12 + 24 == agent.costs['compute_cost']
    candidate, = result['candidates']
    assert candidate['research_program']['source_ref'] == reference
    assert candidate['research_program']['reused_procedure']
    assert candidate['method_source_refs'] == [reference]
    assert result['comparison_completed'] and result['provider_calls'] == 0


@pytest.mark.parametrize('wrong_output', [False, True])
def test_unrelated_parent_or_wrong_graph_output_fails_before_scientific_work(tmp_path, wrong_output):
    agent = agent_with_records(tmp_path)
    parent, reference = retained_procedure(tmp_path)
    if wrong_output:
        saved = json.loads((tmp_path / 'saved_parent_procedure.json').read_text())
        saved['candidates'][0]['research_program']['graph']['nodes']['answer']['answer'] = (
            Program('const', value=99.).to_dict())
        (tmp_path / 'saved_parent_procedure.json').write_text(json.dumps(saved))
    else:
        parent = Program('feature', name='u')
    question = {'id': 'revision:actual', 'kind': 'program_revision'}
    with pytest.raises(ValueError, match='actual parent|differs from its parent'):
        run(agent, question, parent_programs={'actual': parent},
            retained_research_program_refs=[reference])
    receipt = question['native_representation_research'][-1]
    assert receipt['status'] == 'native_representation_execution_failed'
    assert receipt['compute_cost'] == agent.costs['compute_cost'] == 0
    assert receipt['selected'] is None and not receipt['candidates']


def test_replay_does_not_displace_four_original_grammar_candidates(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    parent, reference = retained_procedure(tmp_path)

    def compose(self, spec, rows, budget, parents, actions, *, replayed):
        assert self.config.max_candidates == 4
        return [*replayed, *[{'candidate': Candidate(str(index),
            Program('const', value=float(index)), 'test'), 'training_loss': float(index)}
            for index in range(10, 14)]]

    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', compose)
    result = run(agent, {'id': 'revision:actual', 'kind': 'program_revision'},
        parent_programs={'actual': parent}, retained_research_program_refs=[reference])
    assert len(result['candidates']) == 5
    assert {candidate['program']['value'] for candidate in result['candidates']
            if candidate['program']['op'] == 'const'} == {10., 11., 12., 13.}
    assert result['comparison_completed']
    assert result['compute_cost'] == 60 + 144 + 5 * 12 + 24


def test_failed_replay_is_retained_with_paid_cost_and_source(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    parent, reference = retained_procedure(tmp_path)
    question = {'id': 'revision:actual', 'kind': 'program_revision'}

    def broken(payload, rows, budget, **kwargs):
        budget.spend(7)
        raise ValueError('saved procedure failed on current training')

    monkeypatch.setattr('hegel_machine.cognition.research_programs.replay_research_program', broken)
    with pytest.raises(ValueError, match='saved procedure failed'):
        run(agent, question, parent_programs={'actual': parent},
            retained_research_program_refs=[reference])
    receipt = question['native_representation_research'][-1]
    complete = load_native_representation_result(receipt['source_ref'], agent.workspace)
    replay, = complete['actions']
    assert replay['status'] == 'execution_failed' and replay['source_ref'] == reference
    assert replay['compute_cost'] == 7
    assert complete['compute_cost'] == 60 + 7


def test_generic_duplicate_of_replay_exports_one_program_with_actual_source(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    parent, reference = retained_procedure(tmp_path)

    def compose(self, spec, rows, budget, parents, actions, *, replayed):
        duplicate = dict(replayed[0])
        duplicate.pop('method_source_refs')
        return [*replayed, duplicate]

    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', compose)
    result = run(agent, {'id': 'revision:actual', 'kind': 'program_revision'},
        parent_programs={'actual': parent}, retained_research_program_refs=[reference])
    candidate, = result['candidates']
    assert candidate['method_source_refs'] == [reference]
    assert result['compute_cost'] == 60 + 144 + 12 + 24


def test_no_executable_replay_keeps_negative_result_without_fitting_validation(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    parent, reference = retained_procedure(tmp_path)

    def none(payload, rows, budget, **kwargs):
        assert len(rows) == 48
        budget.spend(7)
        return None

    def compose(self, spec, rows, budget, parents, actions, *, replayed):
        assert replayed == []
        return []

    monkeypatch.setattr('hegel_machine.cognition.research_programs.replay_research_program', none)
    monkeypatch.setattr(UnifiedResearchAgent, '_program_claims', compose)
    result = run(agent, {'id': 'revision:actual', 'kind': 'program_revision'},
        parent_programs={'actual': parent}, retained_research_program_refs=[reference])
    replay, = result['actions']
    assert replay['status'] == 'no_executable_refit' and replay['compute_cost'] == 7
    assert result['status'] == 'native_representation_parent_retained'
    assert result['compute_cost'] == 60 + 7 + 24


@pytest.mark.parametrize('operation', ['window_sum', 'accumulate'])
def test_same_variable_competing_program_opens_coordinates_without_failure_count(tmp_path, operation):
    agent = agent_with_records(tmp_path)
    question = {'id': 'revision:competing', 'kind': 'program_revision',
        'representation_research_basis': competing_representation_basis(agent, tmp_path, operation)}
    result = run(agent, question)
    assert result['alternative_representation_basis'] == 'retained_alternative_has_different_task_predictions'
    assert [item['direction'] for item in result['search_directions']] == ['coordinates', 'retained_composition']
    assert result['candidates'] and result['selected']['changed_prediction_rows'] > 0
    assert result['provider_calls'] == result['new_observations'] == 0


def test_bound_unresolved_semantics_opens_coordinates_without_failures(tmp_path):
    agent = agent_with_records(tmp_path)
    uses = [{'kind': 'supply_component', 'model': 'parent.py', 'slot': 'representation'}]
    agent.questions.append({'id': 'history_semantics', 'meaning_version': 2,
        'kind': 'representation', 'status': 'open', 'use_sites': uses,
        'answer_resolution_ref': {'evaluator': 'representation_diagnostic_and_paid_comparison',
            'parent_model': 'parent.py'}})
    question = {'id': 'revision:semantics', 'kind': 'program_revision', 'use_sites': uses,
        'representation_research_basis': {'semantic_question_ref': {
            'kind': 'question', 'id': 'history_semantics', 'version': 2}}}
    result = run(agent, question)
    assert result['alternative_representation_basis'] == 'unresolved_bound_representation_semantics'
    assert result['candidates'] and result['comparison_completed']


def test_actual_learned_method_reference_is_an_independent_research_warrant(tmp_path):
    from test_question_routing import ROLES
    agent = agent_with_records(tmp_path)
    question = {'id': 'revision:learned', 'kind': 'program_revision',
        'theory_target': {'role_signature': ROLES, 'output_unit': 'V'},
        'representation_research_basis': learned_representation_basis(tmp_path)}
    result = run(agent, question)
    assert result['alternative_representation_basis'] == 'retained_learned_method_proposes_representation_research'
    assert result['candidates'] and result['comparison_completed']


def test_two_claims_on_the_same_failed_rows_do_not_make_continuing_failures(tmp_path, monkeypatch):
    agent = agent_with_records(tmp_path)
    basis = measured_failure_basis(agent, Program('window_sum', lag=3,
        children=(Program('feature', name='u'),)))
    agent.questions[-1]['semantic_answer']['evidence'] = agent.questions[-2]['semantic_answer']['evidence']
    def unexpected(*args):
        raise AssertionError('one witnessed failure was counted as continuing failures')
    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', unexpected)
    result = run(agent, {'id': 'revision:one_failure', 'kind': 'program_revision',
        'representation_research_basis': basis})
    assert result['alternative_representation_basis'] is None


def test_missing_optional_reference_does_not_block_another_actual_or_warrant(tmp_path):
    agent = agent_with_records(tmp_path)
    basis = measured_failure_basis(agent, Program('window_sum', lag=3,
        children=(Program('feature', name='u'),)))
    basis['alternative_ref'] = {'file': str(tmp_path / 'missing.json'), 'json_path': []}
    result = run(agent, {'id': 'revision:or', 'kind': 'program_revision',
        'representation_research_basis': basis})
    assert result['alternative_representation_basis'] == 'prospective_failures_with_public_unrepresented_distinction'



def test_saved_parameter_refit_method_does_not_claim_a_representation_warrant(tmp_path, monkeypatch):
    from test_question_routing import ROLES
    agent = agent_with_records(tmp_path)
    def unexpected(*args):
        raise AssertionError('same-coordinate coefficient refit forced independent history search')
    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims', unexpected)
    result = run(agent, {'id': 'revision:parameter_method', 'kind': 'program_revision',
        'theory_target': {'role_signature': ROLES, 'output_unit': 'V'},
        'representation_research_basis': learned_representation_basis(tmp_path, parameter_only=True)})
    assert result['alternative_representation_basis'] is None
    assert result['comparison_completed']



def test_unlabelled_representation_research_uses_actual_conflict_origin_only_with_witnesses(tmp_path):
    agent = agent_with_records(tmp_path)
    diagnostic = representation_ambiguity([0.] * len(agent.rows), agent.rows)
    assert diagnostic['conflicting_groups'] > 0
    result = run(agent, {'id': 'representation:witness', 'kind': 'representation',
        'representation_diagnostic': diagnostic})
    assert result['origin'] == 'computed_representation_conflict'
