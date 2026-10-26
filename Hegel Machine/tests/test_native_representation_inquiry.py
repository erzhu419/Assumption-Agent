"""Native representation research consumes only its declared training split."""

from dataclasses import replace
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
    assert 'status' not in question  # Installation and child resolution belong to the caller.


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
        raise AssertionError('program revision searched coordinates without a collision')

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


def retained_procedure(tmp_path, *, intercept=2., slope=3.):
    """A saved actual-parent graph, with no fitting in the test fixture."""
    from hegel_machine.cognition.continuous_state import _sum_terms
    from hegel_machine.cognition.research_programs import _entry
    term = Program('feature', name='u')
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
