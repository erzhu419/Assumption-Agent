"""Component support uses the task's MSE contract during fit and later checks."""

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.observable_research import _update_support
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from test_common_archive_observable_research import MeasuredLab
from test_inquiry_revision import action


@pytest.mark.parametrize('kind', ['observable', 'conditional'])
@pytest.mark.parametrize('threshold', [1e-10, 1e-6])
def test_same_task_tolerance_controls_initial_and_later_component_support(tmp_path, kind, threshold):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=10, confirmation_rows=1,
            execution_seconds=30., compute_budget=100, confirmation_mse_threshold=threshold))
    lab = MeasuredLab()
    agent._execute('query', {'actions': [action(0, channels=['meter'])]}, lab)
    if kind == 'observable':
        arguments = {'target': 'meter', 'model': 'model.py',
                     'code': "def predict(row): return 2*row['features']['u']+1.0005\n"}
        entries, name = agent.observable_models, 'meter'
    else:
        agent._execute('write', {'path': 'view.py', 'code': '''def bind(spec,observations,memory):
    return {'rows': [dict(row,evidence=[index]) for record in observations
        for index,row in zip(record['row_indices'],record['rows'])]}
'''}, lab)
        arguments = {'component': 'relation', 'view': 'view.py', 'model': 'model.py',
                     'code': "def predict(row): return 7*row['features']['u']+3.0005\n"}
        entries, name = agent.conditional_models, 'relation'
    evaluated = agent._execute('evaluate', arguments, lab)
    assert evaluated['mse'] == pytest.approx(2.5e-7)
    assert entries[name]['support'] == ('paid_fit' if threshold == 1e-6 else 'refuted_on_paid_rows')
    assert entries[name]['mse_threshold'] == threshold and entries[name]['loss_kind'] == 'mse'
    frozen = entries[name]['model']
    agent._execute('query', {'actions': [action(1, channels=['meter'])]}, lab)
    expected = 'supported_on_later_readings' if threshold == 1e-6 else 'refuted_on_paid_rows'
    assert entries[name]['support'] == expected
    # Reassessing an explicitly chosen retained predictor uses the same contract.
    agent._execute('evaluate', {key: value for key, value in {
        **arguments, 'model': frozen}.items() if key not in {'view', 'code'}}, lab)
    assert entries[name]['support'] == expected and entries[name]['model'] == frozen


def test_direct_legacy_support_helper_keeps_its_default_threshold():
    entry = {'training_observation_indices': [0]}
    _update_support(entry, {'returncode': 0, 'mse': 2.5e-7}, [0])
    assert entry['support'] == 'refuted_on_paid_rows' and entry['mse_threshold'] == 1e-10
