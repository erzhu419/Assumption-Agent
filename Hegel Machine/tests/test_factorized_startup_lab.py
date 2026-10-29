"""Independent task factors preserve the public interface and old learners."""

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab, CONTROL_LEVELS
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.state_search import closed_state_claims, closed_startup_claims


def action(*values):
    return {'kind': 'trajectory', 'sequence': [{'u': value} for value in values], 'channels': []}


def targets(lab, *values):
    return [row['target'] for row in lab.execute(action(*values))['rows']]


@pytest.mark.parametrize('seed', [31, 79, 113])
def test_equal_factor_seeds_exactly_reproduce_historical_apparatus(seed):
    old = AnonymousHistoryLab('hysteresis', seed)
    new = AnonymousHistoryLab('factorized_hysteresis', seed, startup_seed=seed)
    assert new.public_spec() == old.public_spec()
    assert new.task_interface == old.task_interface
    for force in (-1., 1.):
        for value in CONTROL_LEVELS:
            assert new.execute(action(force, value))['rows'] == old.execute(action(force, value))['rows']
    assert new.evaluation_rows(length=110, seed=733) == old.evaluation_rows(length=110, seed=733)


def test_startup_and_mature_factors_have_separate_causal_effects():
    baseline = AnonymousHistoryLab('factorized_hysteresis', 31, startup_seed=31)
    startup = AnonymousHistoryLab('factorized_hysteresis', 31, startup_seed=79)
    mature = AnonymousHistoryLab('factorized_hysteresis', 79, startup_seed=31)
    first = [targets(baseline, value) for value in CONTROL_LEVELS]
    assert first != [targets(startup, value) for value in CONTROL_LEVELS]
    assert first == [targets(mature, value) for value in CONTROL_LEVELS]
    # A forcing control fixes the same observed state before testing the next
    # transition. Startup can affect subsequent states but not this rule.
    old_mature, changed_mature = [], []
    for force in (-1., 1.):
        for value in CONTROL_LEVELS:
            before = targets(baseline, force, value)
            assert before == targets(startup, force, value)
            old_mature.append(before[-1])
            changed_mature.append(targets(mature, force, value)[-1])
    assert old_mature != changed_mature


def test_new_family_resets_and_keeps_parameters_out_of_public_inputs():
    lab = AnonymousHistoryLab('factorized_hysteresis', 31, startup_seed=79)
    query = action(.3, .5, -.1, .7)
    first = lab.execute(query)
    lab.execute(action(1., -1., 1.))
    assert first['rows'] == lab.execute(query)['rows']
    expected = [(row.features, row.history) for row in lab.prediction_rows(query)]
    assert expected == [(row.features, row.history) for row in lab.observation_rows(first)]
    assert set(lab.public_spec()) == set(AnonymousHistoryLab().public_spec())
    before = lab.costs
    measured = lab.evaluation_sequences([query['sequence']])
    assert [row.target for row in measured] == [row['target'] for row in first['rows']]
    assert lab.costs == before


def test_existing_full_and_local_research_learn_from_factorized_measurements():
    def acquire(lab):
        return [lab.execute(action(value)) for value in CONTROL_LEVELS] + [
            lab.execute(action(force, value))
            for force in (-1., 1.) for value in CONTROL_LEVELS]

    original = AnonymousHistoryLab('hysteresis', 31)
    old = closed_state_claims(acquire(original), budget=SearchBudget(30000), max_candidates=4)
    programs = [item['candidate'].program for item in old if item['training_loss'] == 0]
    assert programs
    changed = AnonymousHistoryLab('factorized_hysteresis', 31, startup_seed=79)
    records = acquire(changed)
    rows = [row for record in records for row in changed.observation_rows(record)]
    local = closed_startup_claims(programs, rows, budget=SearchBudget(30000), max_candidates=4)
    full = closed_state_claims(records, budget=SearchBudget(30000), max_candidates=4)
    probe = changed.evaluation_rows(length=128, seed=733)
    for candidates in (local, full):
        exact = [item['candidate'].program for item in candidates if item['training_loss'] == 0]
        assert exact
        assert any(all(program.predict(row) == row.target for row in probe) for program in exact)
    assert all(item['state_research']['mature_transition_preserved'] for item in local)


def test_case_requires_explicit_independent_startup_factor():
    with pytest.raises(ValueError, match='independent startup seed'):
        AnonymousHistoryLab('factorized_hysteresis', 31)
    with pytest.raises(ValueError, match='belongs to the factorized'):
        AnonymousHistoryLab('hysteresis', 31, startup_seed=79)
