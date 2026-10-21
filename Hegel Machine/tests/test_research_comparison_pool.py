"""Research diversity uses paid forecasts and does not become task adoption."""

from hegel_machine.cognition.candidate_comparison import compare_candidate_adoption
from hegel_machine.cognition.evaluation import predict
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.selection import (
    rank_candidates, retain_research_comparison_pool, screen_candidates,
)


def fixture():
    x, z = Program('feature', name='x'), Program('feature', name='z')
    candidates = [Candidate('old_x', x, 'reuse', memory_id='old_source'),
                  Candidate('old_z', z, 'refit'),
                  Candidate('square', Program('mul', children=(x, x)), 'composition')]
    observations = [Observation({'x': value, 'z': value}, value) for value in (0., 1.)]
    public_rows = [Observation({'x': 2., 'z': 2.}, 999.)]
    return candidates, observations, public_rows


def paid_public_forecasts(assessed, public_rows, budget):
    predictions = {}
    for row in assessed:
        program = row.candidate.program.expanded()
        if program not in predictions:
            predictions[program] = (tuple(predict(program, point, 0.) for point in public_rows)
                                    if budget.spend(len(public_rows)) else None)
    return predictions


def test_task_unadmitted_disagreement_enters_research_pool_without_changing_native_adoption():
    candidates, observations, public_rows = fixture()
    assessment, work = [], SearchBudget(6)
    task = screen_candidates(candidates, observations, [], fallback=0., q=.5, budget=work,
        complexity_weight=.001, maximum=2, assessment_sink=assessment)
    assert [row.candidate.candidate_id for row in task] == ['old_x', 'old_z']
    assert len(assessment) == 3 and work.spent == 6
    fresh = [Observation({'x': 2., 'z': -2.}, 2.)]
    comparison = rank_candidates([row.candidate for row in task], observations, fresh,
        fresh=fresh, fallback=0., q=.5, budget=SearchBudget(8), complexity_weight=.001)
    kwargs = {'fallback': 0., 'spec': TaskSpec('test', 'Predict the observed response', ('x', 'z'), (), ()),
              'config': ResearchConfig(selection_policy='prospective')}
    before = compare_candidate_adoption(comparison, candidates[0], observations, [], fresh, **kwargs)
    forecasting = SearchBudget(3)
    table = paid_public_forecasts(assessment, public_rows, forecasting)
    research, details = retain_research_comparison_pool(assessment, table, maximum=2)
    assert [row.candidate.candidate_id for row in research] == ['old_x', 'square']
    assert 'different_public_behavior' in details['retained'][1]['reasons']
    assert forecasting.spent == 3 and work.spent == 6
    after = compare_candidate_adoption(comparison, candidates[0], observations, [], fresh, **kwargs)
    assert before == after and not after.adopt and after.current.candidate == candidates[0]
    assert [row.candidate.candidate_id for row in task] == ['old_x', 'old_z']
    assert not details['task_adoption_evaluated']


def test_expanded_definition_clones_share_one_research_slot_and_keep_all_origins(monkeypatch):
    candidates, observations, public_rows = fixture()
    argument = Program('arg', name='value')
    identity = Definition('independently_named_identity', ('value',), argument)
    clone = Candidate('new_label', Program('call', name=identity.definition_id,
        children=(candidates[0].program,), definition=identity), 'new_object', memory_id='other_source')
    assessment = rank_candidates([candidates[0], clone, *candidates[1:]], observations, [],
        fallback=0., q=.5, budget=SearchBudget(8), complexity_weight=.001)
    work = SearchBudget(3)
    table = paid_public_forecasts(assessment, public_rows, work)

    def no_free_prediction(*args, **kwargs):
        raise AssertionError('Research retention must reuse the charged prediction table')

    monkeypatch.setattr(Program, 'predict', no_free_prediction)
    research, details = retain_research_comparison_pool(assessment, table, maximum=2)
    assert [row.candidate.candidate_id for row in research] == ['old_x', 'square']
    assert details['input_candidate_count'] == 4 and details['unique_program_count'] == 3
    assert {origin.candidate_id for origin in research[0].origins} == {'old_x', 'new_label'}
    assert details['expanded_clone_groups'][0]['origin_ids'] == ['old_x', 'new_label']
    assert len(assessment[0].origins) == 1 and work.spent == 3


def test_unfunded_public_forecast_is_unknown_and_does_not_gain_a_behavior_slot():
    candidates, observations, public_rows = fixture()
    assessment = rank_candidates(candidates, observations, [], fallback=0., q=.5,
        budget=SearchBudget(6), complexity_weight=.001)
    forecasting = SearchBudget(2)
    table = paid_public_forecasts(assessment, public_rows, forecasting)
    research, details = retain_research_comparison_pool(assessment, table, maximum=2)
    assert forecasting.spent == 2 and forecasting.denied_requests == 1
    assert [row.candidate.candidate_id for row in research] == ['old_x', 'old_z']
    assert details['unknown_prediction_ids'] == ['square']
    assert len(details['public_behavior_groups']) == 1
    assert all('different_public_behavior' not in item['reasons'] for item in details['retained'])


def test_assessment_sink_keeps_the_real_sample_scope_without_changing_task_screen_or_cost():
    candidates, observations, _ = fixture()
    observations *= 12
    ordinary_work, research_work = SearchBudget(36), SearchBudget(36)
    kwargs = {'fallback': 0., 'q': .5, 'complexity_weight': .001, 'maximum': 2}
    ordinary = screen_candidates(candidates, observations, [], budget=ordinary_work, **kwargs)
    assessment = []
    task = screen_candidates(candidates, observations, [], budget=research_work,
                             assessment_sink=assessment, **kwargs)
    assert [row.candidate for row in task] == [row.candidate for row in ordinary]
    assert research_work.spent == ordinary_work.spent == 36
    assert len(assessment) == 3
    for row in assessment:
        assert row.assessment_scope['kind'] == 'sampled_evidence'
        assert row.assessment_scope['observation_count'] == 24
        assert len(row.assessment_scope['assessed_row_indices']) == 12
    research, details = retain_research_comparison_pool(assessment, {}, maximum=2)
    assert details['retained'][0]['assessment_scope'] == research[0].assessment_scope
    assert set(details['unknown_prediction_ids']) == {'old_x', 'old_z', 'square'}
