"""Paid closed predictions distinguish initial failure from later propagation."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_conditions import paid_transition_evidence, _method_observables
from hegel_machine.cognition.method_feedback import begin_method_feedback, retained_transition_evidence
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_python_paid_score_reuse import scene, choose, no_execution


def columns():
    rows, records = [], []
    for controls in ((0., 1., 2.), (0., 1., 2.)):
        start = len(rows)
        for position, value in enumerate(controls):
            rows.append(Observation({'u': value}, 0., tuple({'u': x} for x in controls[:position])))
        records.append({'rows': [asdict(row) for row in rows[start:]],
                        'row_indices': list(range(start, len(rows)))})
    return rows, records


def test_first_error_after_correct_prefix_is_separate_from_propagation():
    rows, records = columns()
    result = paid_transition_evidence(records, rows, [1., 1., 1., 0., 1., 1.], guard_depth=1)
    assert result['startup_first_divergences'] == result['mature_first_divergences'] == 1
    assert result['correct_prefix_first_divergences'] == 1
    assert result['later_errors_unattributed'] == 3
    assert result['error_rows'] == 5
    # A two-step startup guard must not turn a second-row failure into a
    # mature-transition refutation.
    deeper = paid_transition_evidence(records, rows, [1., 1., 1., 0., 1., 1.], guard_depth=2)
    assert deeper['startup_first_divergences'] == 2
    assert deeper['mature_first_divergences'] == 0


@pytest.mark.parametrize('missing', ['record', 'prefix', 'predictions', 'record_rows'])
def test_partial_or_mismatched_paid_view_is_unknown_not_zero(missing):
    rows, records = columns()
    predictions = [0.] * len(rows)
    if missing == 'record':
        records.pop()
    elif missing == 'prefix':
        records[0]['rows'][0]['history'] = [{'u': -1.}]
    elif missing == 'predictions':
        predictions.pop()
    else:
        records[0]['rows'][0]['target'] = 1.
    result = paid_transition_evidence(records, rows, predictions, guard_depth=1)
    assert result['status'] == 'unknown'
    assert 'mature_first_divergences' not in result


def test_unexposed_guard_does_not_claim_mature_correctness():
    rows, records = columns()
    result = paid_transition_evidence(records, rows, [0., 1., 1., 0., 0., 0.])
    assert result['status'] == 'observed'
    assert result['correct_prefix_first_divergences'] == 1
    assert result['mature_first_divergences'] is None
    assert result['startup_first_divergences'] is None


@pytest.mark.parametrize('correct_start', [False, True])
def test_profile_reads_current_paid_score_and_measured_response_history_only(tmp_path, monkeypatch, correct_start):
    agent, lab = scene(tmp_path)
    model = Program('if', children=(Program('available', lag=1), Program('const', value=0.),
        Program('feature', name='u') if correct_start else Program('const', value=0.)))
    agent._save('retained.py', export_temporal_program(model))
    original_rows = deepcopy(agent.rows)
    choose(agent, lab, ['retained.py'])
    agent.theory_consolidation.setdefault('candidates', {})['fixture'] = {'kind': 'research_program'}
    costs = deepcopy(agent.costs)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    monkeypatch.setattr(agent, '_score', no_execution)
    profile = begin_method_feedback(agent, lab)['profile']
    transition = profile['residual']['transition']
    assert transition['status'] == 'observed'
    assert transition['startup_first_divergences'] == int(not correct_start)
    assert transition['mature_first_divergences'] == int(correct_start)
    assert transition['later_errors_unattributed'] == int(not correct_start)
    assert {item['condition'] for item in profile['data']['conditional_coverage']} == {
        'startup', 'previous_response_0'}
    assert agent.rows == original_rows and agent.costs == costs
    assert all('__target__' not in point for row in agent.rows for point in row.history)
    assert len(profile['transition_evidence_refs']) == 2
    coordinates = _method_observables(profile)
    assert ('residual', 'transition', 'mature_first_divergences') in coordinates
    assert not any('source_refs' in path or 'transition_evidence_refs' in path for path in coordinates)


@pytest.mark.parametrize('changed', ['target', 'public_spec', 'retained_source', 'new_rows'])
def test_old_score_cannot_diagnose_changed_rows_or_current_model(tmp_path, changed):
    agent, lab = scene(tmp_path)
    agent._save('retained.py', export_temporal_program(Program('const', value=0.)))
    choose(agent, lab, ['retained.py'])
    public = lab.public_spec()
    if changed == 'target':
        agent.rows[0] = replace(agent.rows[0], target=3.)
    elif changed == 'public_spec':
        public['max_probe_length'] = 99
    elif changed == 'retained_source':
        (agent.workspace / agent._evaluated_summary()['model']).write_text(
            export_temporal_program(Program('const', value=3.)))
    else:
        agent.rows.append(Observation({'u': 3.}, 3.))
    result = retained_transition_evidence(agent, public)
    assert result['status'] == 'unknown'
    assert 'mature_first_divergences' not in result


def test_ordinary_authored_predictor_remains_valid_without_closed_prediction_diagnostic(tmp_path):
    agent, lab = scene(tmp_path)
    agent._save('authored.py', 'def predict(row): return 0.\n')
    choose(agent, lab, ['authored.py'])
    agent.theory_consolidation.setdefault('candidates', {})['fixture'] = {'kind': 'research_program'}
    profile = begin_method_feedback(agent, lab)['profile']
    assert profile['residual']['loss'] == agent._evaluated_summary()['mse']
    assert 'transition' not in profile['residual']
