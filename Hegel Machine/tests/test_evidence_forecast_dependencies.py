"""New child inputs reach forecasts before their frozen experiment is purchased."""

import pytest

from test_executable_evidence_binding import setup, return_region
from test_inquiry_revision import action, revision


@pytest.mark.parametrize('arm', ['hegel', 'code_react'])
@pytest.mark.parametrize('path', ['comparison', 'direct_revision'])
def test_new_child_version_reaches_current_forecasts_without_rewriting_flat_choice(tmp_path, arm, path):
    agent, lab = setup(tmp_path, arm)
    return_region(agent, lab, 0., 1.)
    agent._execute('evaluate', {'model': 'parent.py',
        'research_inputs': {'slope': 'slope_question'}}, lab)
    return_region(agent, lab, .9, 1.1)
    assert agent.research_candidates['parent.py']['status'] == 'needs_rebuild'
    if path == 'comparison':
        compared = agent._execute('run', {'operator': 'compare_experiments',
            'models': ['parent.py'], 'experiments': [
                {'id': 'contrast', 'actions': [action(2)]}]}, lab)['comparison']
        forecast = compared['forecasts'][0]
        update = compared['candidate_contexts'][0]['research_input_update']
        assert lab.paid == 4
    else:
        proposed = agent._execute('query', {'actions': [action(2)],
            'revision': revision(['parent.py'])}, lab)['revision']
        forecast = proposed['forecasts'][0]
        update = forecast['research_input_update']
        assert lab.paid == 5
    assert forecast['predictions'] == ([2.] if arm == 'hegel' else [1.])
    binding = forecast['research_bindings'][0]
    assert binding['inputs']['slope']['version'] == (2 if arm == 'hegel' else 1)
    if arm == 'code_react':
        assert update['status'] == 'old_dependency_version'
    else:
        assert update['inputs']['slope']['version'] == 2
