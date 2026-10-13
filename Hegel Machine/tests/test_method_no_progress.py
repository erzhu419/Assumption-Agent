"""Completed empty research cycles return to their author without choosing science."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_method import _comparison_state
from scripts.experimental_design_control import ExperimentalDesignMethodAgent
from test_experimental_design_control import prepared
from test_inquiry_revision import action, write


def method(body):
    return ("import json\ndef select_actions(spec, records, memory):\n"
            "    step = len(json.load(open('research_state.json'))['method_results'])\n" + body)


def test_same_completed_comparison_returns_paid_feedback_after_second_execution(tmp_path):
    agent, lab, arguments = prepared(tmp_path, ExperimentalDesignMethodAgent)
    source = method(f"    return {{'action':'run','arguments':{arguments!r}}}\n")
    agent.bind_method(source)
    checkpoints = []
    result = agent.execute_method(lab, on_checkpoint=checkpoints.append)
    assert result['status'] == 'method_no_progress'
    assert len(result['method_results']) == len(result['experiment_comparisons']) == 2
    first, second = result['experiment_comparisons']
    assert first['candidate_contexts'] != second['candidate_contexts']
    assert first['index'] != second['index']
    assert result['no_progress']['comparison_indices'] == [0, 1]
    assert result['no_progress']['evidence_cutoff'] == {'records': 1, 'rows': 1}
    assert result['method_results'][-1]['feedback']['no_progress'] == result['no_progress']
    assert checkpoints[-1]['no_progress'] == result['no_progress']
    assert lab.paid == result['costs']['observation_cost'] == 1
    assert result['costs']['compute_cost'] == first['compute_cost'] + second['compute_cost']
    assert result['costs']['model_calls'] == 0 and result['model'] is None
    assert (tmp_path / METHOD_PATH).read_text() == source


@pytest.mark.parametrize('change', ['source', 'predictions', 'observations', 'experiments'])
def test_changed_candidate_evidence_or_experiment_does_not_end_research(tmp_path, change):
    agent, lab, arguments = prepared(tmp_path, ExperimentalDesignMethodAgent)
    for name, coefficient in [('a.py', 1.), ('b.py', 0.)]:
        write(agent, name, f"def predict(row): return {coefficient} * row['features']['u']\n", lab)
    next_arguments = deepcopy(arguments)
    if change == 'source':
        # Even an equal prediction on this panel does not erase a source revision.
        middle = {'action': 'write', 'arguments': {'path': 'b.py',
                  'code': 'def predict(row): return 0. * row["features"]["u"]\n'}}
    elif change == 'predictions':
        write(agent, 'scale.txt', '0', lab)
        write(agent, 'b.py', "def predict(row): return float(open('scale.txt').read()) * row['features']['u']\n", lab)
        middle = {'action': 'write', 'arguments': {'path': 'scale.txt', 'code': '2'}}
    elif change == 'observations':
        middle = {'actions': [action(.25)]}
    else:
        next_arguments['experiments'][1]['actions'] = [action(.5)]
        middle = {'action': 'run', 'arguments': {'code': 'pass'}}
    agent.bind_method(method(
        f"    if step == 0: return {{'action':'run','arguments':{arguments!r}}}\n"
        f"    if step == 1: return {middle!r}\n"
        f"    if step == 2: return {{'action':'run','arguments':{next_arguments!r}}}\n"
        "    return {'stop':'The planned comparisons are complete.'}\n"))
    result = agent.execute_method(lab)
    assert result['status'] == 'method_stopped' and 'no_progress' not in result
    assert len(result['experiment_comparisons']) == 2
    first, second = result['experiment_comparisons']
    assert _comparison_state(first) != _comparison_state(second)
    assert len(result['method_results']) == 4


@pytest.mark.parametrize('recover', [False, True])
def test_first_error_can_be_repaired_but_same_error_twice_returns_to_author(tmp_path, recover):
    agent, lab, _ = prepared(tmp_path, ExperimentalDesignMethodAgent)
    failed = {'action': 'run', 'arguments': {'code': "raise ValueError('unfinished analysis')"}}
    body = ("    if step == 1: return {'action':'run','arguments':{'code':'value = 1'}}\n"
            "    if step == 2: return {'stop':'Analysis repaired.'}\n") if recover else ''
    agent.bind_method(method(body + f"    return {failed!r}\n"))
    result = agent.execute_method(lab)
    assert result['costs']['model_calls'] == 0 and lab.paid == 1
    if recover:
        assert result['status'] == 'method_stopped' and 'no_progress' not in result
        assert len(result['method_results']) == 3
    else:
        assert result['status'] == 'method_no_progress'
        assert len(result['method_results']) == 2
        assert result['no_progress']['kind'] == 'repeated_execution_error'
        assert result['no_progress']['failure']['stderr'] == ['ValueError: unfinished analysis']


def test_legitimate_multi_step_analysis_without_observations_keeps_running(tmp_path):
    agent, lab, _ = prepared(tmp_path, ExperimentalDesignMethodAgent)
    agent.bind_method(method(
        "    if step < 5: return {'action':'run','arguments':{'code':'pass'}}\n"
        "    return {'stop':'Finished planned computation.'}\n"))
    result = agent.execute_method(lab)
    assert result['status'] == 'method_stopped' and len(result['method_results']) == 6
    assert lab.paid == 1 and 'no_progress' not in result
