"""Paid probes retain their individual evidence and original readout fitting scope."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.transfer_partitions import continue_transfer_partition
from test_transfer_partitions import second_stage_fixture


def test_real_feedback_groups_preserve_history_errors_and_original_fit_scope(monkeypatch):
    import hegel_machine.cognition.research_programs as module
    entry, training, feedback, measured = second_stage_fixture()
    original = deepcopy(entry)
    groups = [deepcopy(feedback[:2]), deepcopy(feedback[2:])]
    native, calls = RelationIntervalResearch.alternatives, []

    def observed(self, paid, budget, tolerance, **kwargs):
        parts = kwargs['experiments']
        assert [len(part) for part in parts] == [2, 6]
        assert all(actual is old for actual, old in zip(
            [row for part in parts for row in part], feedback))
        assert all(actual.history == old.history for actual, old in zip(paid, training + feedback))
        calls.append(parts)
        return native(self, paid, budget, tolerance, **kwargs)

    monkeypatch.setattr(RelationIntervalResearch, 'alternatives', observed)
    monkeypatch.setattr(module._Fitter, 'fit', lambda *args: pytest.fail('fixed readout was refitted'))
    source = {'file': 'paid-original-fit.json', 'json_path': [0]}
    budget = SearchBudget(15000)
    result = continue_transfer_partition(entry, training, feedback, budget,
        tolerance=1e-7, feedback_loss=measured, feedback_experiments=groups,
        fit_source_ref=source)
    assert calls and result['status'] == 'paid_interval_supported'
    assert result['compute_cost'] == budget.spent > 0 and result['new_fits'] == 0
    assert result['interval_compute_cost'] == budget.spent
    assert entry == original
    original_fit = research_program_fit_contract(entry['research_program'])
    for candidate in result['candidates']:
        body = candidate['research_program']
        fit = research_program_fit_contract(body)
        errors = candidate['paid_row_squared_errors']
        actual = [(candidate['candidate'].program.predict(row) - row.target) ** 2
                  for row in training + feedback]
        assert errors == actual
        assert len(errors) == candidate['evaluation_rows'] == body['evaluation_rows'] == len(training) + len(feedback)
        assert sum(errors) / len(errors) == pytest.approx(candidate['all_paid_loss'])
        assert body['evaluation_loss'] == candidate['all_paid_loss']
        assert body['fit_rows'] == len(training)
        assert body['readout_fit_source'] == {'source_ref': source, 'fit_rows': len(training),
                                             'fixed_readout': True, 'new_fits': 0}
        assert fit['loss'] == candidate['training_loss']
        assert fit['coefficients'] == original_fit['coefficients']
        assert len(fit['fit_terms']) == len(original_fit['fit_terms'])
        assert all(value <= 1e-7 for value in candidate['feedback_trajectory_losses'])


@pytest.mark.parametrize('cached_errors', [False, True])
def test_small_failed_probe_cannot_be_hidden_by_passing_pooled_mean(monkeypatch, cached_errors):
    import hegel_machine.cognition.parallel_prediction as prediction
    entry, training, feedback, _ = second_stage_fixture()
    values = [entry['candidate'].program.predict(row) for row in feedback]
    feedback = [replace(row, target=value + (.8 if i == 0 else 0.))
                for i, (row, value) in enumerate(zip(feedback, values))]
    errors = [(value - row.target) ** 2 for value, row in zip(values, feedback)]
    assert sum(errors) / len(errors) < .1 < errors[0]
    calls, interval_calls = [], []
    native_predict = prediction.predict_rows

    def counted(program, rows):
        calls.append(list(rows))
        if cached_errors:
            pytest.fail('paid feedback was predicted again')
        return native_predict(program, rows)

    def stop_after_routing(self, rows, budget, tolerance, **kwargs):
        interval_calls.append(kwargs['experiments'])
        self.status = 'refuted'
        self.last_candidate_errors = {}
        return []

    monkeypatch.setattr(prediction, 'predict_rows', counted)
    monkeypatch.setattr(RelationIntervalResearch, 'alternatives', stop_after_routing)
    result = continue_transfer_partition(entry, training, feedback, SearchBudget(15000),
        tolerance=.1, feedback_loss=sum(errors) / len(errors),
        feedback_experiments=[feedback[:1], feedback[1:]],
        feedback_squared_errors=errors if cached_errors else None)
    assert len(calls) == (0 if cached_errors else 1)
    assert interval_calls and result['status'] == 'fixed_readout_interval_unexplained'
    assert result['prior_feedback_trajectory_losses'][0] > .1
    assert result['prior_feedback_trajectory_losses'][1] == 0.


def test_default_feedback_group_keeps_original_pooled_semantics_without_extra_prediction(monkeypatch):
    import hegel_machine.cognition.parallel_prediction as prediction
    entry, training, feedback, _ = second_stage_fixture()
    monkeypatch.setattr(prediction, 'predict_rows', lambda *args: pytest.fail('paid aggregate repeated'))
    result = continue_transfer_partition(entry, training, feedback, SearchBudget(15000),
        tolerance=.1, feedback_loss=.08)
    assert result['status'] == 'paid_return_already_supported' and result['compute_cost'] == 0


def test_feedback_groups_cannot_drop_or_reorder_paid_evidence():
    entry, training, feedback, measured = second_stage_fixture()
    budget = SearchBudget(15000)
    with pytest.raises(ValueError, match='cover all paid feedback'):
        continue_transfer_partition(entry, training, feedback, budget, tolerance=1e-7,
            feedback_loss=measured, feedback_experiments=[feedback[:2]])
    with pytest.raises(ValueError, match='partition the paid feedback in order'):
        continue_transfer_partition(entry, training, feedback, budget, tolerance=1e-7,
            feedback_loss=measured, feedback_experiments=[feedback[2:], feedback[:2]])
    assert budget.spent == 0


@pytest.mark.parametrize('phase', ['construction', 'evaluation'])
def test_nested_interval_cost_survives_activity_interruption_once(monkeypatch, phase):
    entry, training, feedback, measured = second_stage_fixture()
    def stopped_init(self, program, priors, budget, **kwargs):
        budget.spend(7)
        raise TimeoutError('activity deadline')

    def stopped_evaluation(self, rows, budget, tolerance, **kwargs):
        budget.spend(11)
        raise TimeoutError('activity deadline')

    charges = []
    class RecordingBudget(SearchBudget):
        def spend(self, units=1):
            charges.append(units)
            return super().spend(units)

    if phase == 'construction':
        monkeypatch.setattr(RelationIntervalResearch, '__init__', stopped_init)
    else:
        monkeypatch.setattr(RelationIntervalResearch, 'alternatives', stopped_evaluation)
    budget = RecordingBudget(15000)
    with pytest.raises(TimeoutError, match='activity deadline'):
        continue_transfer_partition(entry, training, feedback, budget,
            tolerance=1e-7, feedback_loss=measured)
    assert len(charges) == 1 and budget.spent == charges[0]
    assert budget.spent == 7 if phase == 'construction' else budget.spent >= 11
