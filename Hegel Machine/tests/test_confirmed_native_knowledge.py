"""Actual confirmation can save native knowledge without repeating paid scoring."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.condition_learning import prediction_matches
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import EmpiricalScope, KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, TaskSpec
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def selection():
    x = Program('feature', name='x')
    term = Program('mul', children=(x, x))
    program = Program('add', children=(Program('const', value=.25),
        Program('mul', children=(Program('const', value=1.8), term))))
    rows = [Observation({'x': value}, .25 + 1.8 * value * value)
            for value in (-2., -1., 0., 1., 2.)]
    spec = TaskSpec('paid_source', 'Measured response', ('x',), (), ())
    fit = {'fit_terms': (term,), 'coefficients': [.25, 1.8], 'program': program, 'loss': 0.}
    entry = {'candidate': Candidate('retained_fit', program, 'structure'),
        'kind': 'structural:retained', 'family_id': 'T12', 'training_loss': 0.,
        'validation_loss': 0., 'structure': {'retained_coordinates': True},
        'research_program': _plan(fit, [], len(rows), 123)}
    return entry, spec, rows


def save(agent, entry, spec, rows, **kwargs):
    return agent.remember_confirmed_selection(entry, spec, rows[:3], rows[3:],
        confirmed=True, confirmation_loss=1e-21, confirmation_rows=24,
        selected_channels=('fine',), batches=[{'probe': {'probe_id': 'paid'}, 'channels': ['fine']}],
        confirmation_ref={'file': 'retained/confirmation.json'},
        paid_support_matches=[True] * len(rows), **kwargs)


def test_paid_confirmation_saves_program_and_scope_once_without_predictions(monkeypatch):
    entry, spec, rows = selection()
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))

    def no_science(*args, **kwargs):
        raise AssertionError('Saving paid support must not execute a predictor or fit again')

    monkeypatch.setattr(Program, 'predict', no_science)
    monkeypatch.setattr('hegel_machine.cognition.unified_research.loss', no_science)
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit', no_science)
    result = save(agent, entry, spec, rows)
    assert result['stored'] and result['concepts_added'] == [result['concept_id']]
    prior = agent.library.meta_priors[0]
    assert prior['family_id'] == 'T12' and prior['structure'] == entry['structure']
    assert prior['source_features'] == ['x'] and prior['probe_ids'] == ['paid']
    assert prior['research_program']['status'] == 'supported_on_fresh_trajectory'
    assert prior['research_program']['confirmation_rows'] == 24
    assert prior['research_program']['confirmation_ref'] == {'file': 'retained/confirmation.json'}
    assert prior['research_method']['fit_rows'] == 3
    assert len(agent.library.concepts[0].scopes[0].support_examples) == 5
    frozen = deepcopy(agent.library.to_dict())
    repeated = save(agent, entry, spec, rows)
    assert repeated['reused'] and repeated['concepts_added'] == []
    assert agent.library.to_dict() == frozen


@pytest.mark.parametrize('confirmed,loss,count', [(False, 0., 24), (True, 1., 24), (True, 0., 0)])
def test_working_success_or_failed_confirmation_does_not_install_knowledge(confirmed, loss, count):
    entry, spec, rows = selection()
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    before = deepcopy(agent.library.to_dict())
    result = agent.remember_confirmed_selection(entry, spec, rows[:3], rows[3:],
        confirmed=confirmed, confirmation_loss=loss, confirmation_rows=count)
    assert not result['stored'] and agent.library.to_dict() == before


def test_new_program_fills_existing_concept_without_erasing_old_counterexamples():
    entry, spec, rows = selection()
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    old = deepcopy(entry)
    old.pop('research_program')
    agent.remember_confirmed_selection(old, spec, rows[:3], rows[3:],
        confirmed=True, confirmation_loss=0., confirmation_rows=24)
    prior = agent.library.meta_priors[0]
    prior['status'] = 'conditional'
    prior['counterexamples'].append({'task_id': 'another_task', 'loss': 5.,
                                    'replacement_concept_id': None})
    failures = deepcopy(prior['counterexamples'])
    result = save(agent, entry, spec, rows)
    assert result['stored'] and result['concepts_added'] == []
    assert len(agent.library.concepts) == len(agent.library.meta_priors) == 1
    assert prior['research_program'] == entry['research_program']
    assert prior['status'] == 'conditional' and prior['counterexamples'] == failures


def test_paid_scope_mask_keeps_original_history_and_match_semantics(monkeypatch):
    program = Program('lag', name='x', lag=1)
    rows = [Observation({'x': 9.}, 0.),
            Observation({'x': 2.}, 1., ({'x': 1.},)),
            Observation({'x': 3.}, 100., ({'x': 2.},))]
    spec = TaskSpec('history', 'Retained history', ('x',), (), ())
    matches = [prediction_matches(program, row) for row in rows]
    assert matches == [False, True, False]
    ordinary = EmpiricalScope.measured(spec, program, rows)
    monkeypatch.setattr(Program, 'predict', lambda *a: pytest.fail('Paid scope was predicted again'))
    restored = EmpiricalScope.measured(spec, program, rows, paid_support_matches=matches)
    assert restored == ordinary


def test_existing_scope_filter_keeps_paid_mask_aligned(monkeypatch):
    library = KnowledgeLibrary()
    spec = TaskSpec('filter', 'Paid scope', ('x',), (), ())
    candidate = Candidate('constant', Program('const', value=0.), 'observed')
    initial = [Observation({'x': -2.}, 1.), Observation({'x': -1.}, 1.)]
    identifier, _ = library.learn(candidate, spec, initial, paid_support_matches=[False, False])
    library.concepts[0].scopes[0].conditions = [{'name': 'x', 'lag': 0, 'lower': 0.}]
    rows = [Observation({'x': -1.}, 0.), Observation({'x': 1.}, 1.), Observation({'x': 2.}, 0.)]
    monkeypatch.setattr(Program, 'predict', lambda *a: pytest.fail('Paid target predicted again'))
    same, added = library.learn(candidate, spec, rows, paid_support_matches=[True, False, True])
    assert same == identifier and not added
    assert [row['features']['x'] for row in library.concepts[0].scopes[0].support_examples] == [2.]


def test_misaligned_paid_mask_fails_before_library_mutation():
    entry, spec, rows = selection()
    library = KnowledgeLibrary()
    before = deepcopy(library.to_dict())
    with pytest.raises(ValueError, match='mask must align'):
        library.learn(entry['candidate'], spec, rows, paid_support_matches=[True])
    assert library.to_dict() == before


def test_ordinary_research_uses_the_same_confirmed_save_path(monkeypatch):
    from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab

    agent = UnifiedResearchAgent(ResearchConfig(observation_budget=192,
        compute_budget=120000, minimum_improvement=1e-7, max_candidates=16))
    original, calls = agent.remember_confirmed_selection, []

    def remember(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(agent, 'remember_confirmed_selection', remember)
    result = agent.run(UnknownTaskLab(0, 113))
    assert result['status'] == 'completed'
    assert len(calls) == 1 and calls[0][0]['confirmed']
    assert calls[0][1]['stored'] and result['concepts_added'] == calls[0][1]['concepts_added']
    prior = next(item for item in agent.library.meta_priors
                 if item['source_concept_id'] == result['concepts_added'][0])
    assert prior['validation_loss'] == result['confirmation_loss']
    assert prior['research_method']['confirmation_rows'] > 0
