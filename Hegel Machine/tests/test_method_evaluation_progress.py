"""Unchanged evaluations hand control back without blocking real revisions."""

from dataclasses import replace
import json
import math
import pytest

from scripts.experimental_design_control import ExperimentalDesignMethodAgent
from test_experimental_design_control import prepared
from test_inquiry_revision import action, write
from test_method_no_progress import method


def test_repeated_retained_evaluation_returns_actual_paid_results_to_author(tmp_path):
    agent, lab, _ = prepared(tmp_path, ExperimentalDesignMethodAgent)
    agent.bind_method(method("    return {'action':'evaluate','arguments':{'model':'a.py'}}\n"))
    result = agent.execute_method(lab)
    assert result['status'] == 'method_no_progress'
    assert result['no_progress']['kind'] == 'repeated_candidate_evaluation'
    assert result['no_progress']['steps'] == [0, 1]
    assert result['no_progress']['mse'] == 0.
    assert len(result['method_results']) == 2
    assert result['costs']['compute_cost'] == 2
    assert lab.paid == 1 and result['costs']['model_calls'] == 0
    assert result['method_results'][-1]['feedback']['no_progress'] == result['no_progress']


@pytest.mark.parametrize('revision', ['source', 'dependency', 'observations'])
def test_equal_score_after_real_revision_does_not_trigger_handoff(tmp_path, revision):
    agent, lab, _ = prepared(tmp_path, ExperimentalDesignMethodAgent)
    write(agent, 'scale.txt', '1', lab)
    write(agent, 'a.py', "def predict(row): return float(open('scale.txt').read()) * row['features']['u']\n", lab)
    if revision == 'source':
        # Both source versions give zero on the one purchased row. The method
        # changes the file inside its invocation, before returning evaluate.
        change = "    if step == 1: open('a.py','w').write(\"def predict(row): return 2. * row['features']['u']\\n\")\n"
    elif revision == 'dependency':
        change = "    if step == 1: open('scale.txt','w').write('2')\n"
    else:
        change = f"    if step == 1: return {{'actions': [{action(0.)!r}]}}\n"
    last_evaluation = 2 if revision == 'observations' else 1
    agent.bind_method(method(change +
        f"    if step > {last_evaluation}: return {{'stop':'Completed revised evaluation.'}}\n"
        "    return {'action':'evaluate','arguments':{'model':'a.py'}}\n"))
    result = agent.execute_method(lab)
    assert result['status'] == 'method_stopped' and 'no_progress' not in result
    evaluations = [item for item in result['method_results']
                   if (item['proposal'] or {}).get('action') == 'evaluate']
    assert len(evaluations) == 2
    assert [item['feedback']['mse'] for item in evaluations] == [0., 0.]


def rejected_candidate(path):
    agent, lab, _ = prepared(path, ExperimentalDesignMethodAgent)
    write(agent, 'a.py', "def predict(row): return row['features']['u']\n", lab)
    agent._execute('evaluate', {'model': 'a.py'}, lab)
    write(agent, 'scale.txt', '1', lab)
    write(agent, 'b.py', "def predict(row): return 1. + float(open('scale.txt').read()) * row['features']['u']\n", lab)
    return agent, lab


def test_repeated_rejected_candidate_returns_to_author_without_new_snapshot(tmp_path):
    agent, lab = rejected_candidate(tmp_path)
    incumbent = (tmp_path / '_evaluated_model/a.py').read_bytes()
    candidate = (tmp_path / 'b.py').read_text()
    # Real trial012 rewrites identical exported parameters and source each step.
    agent.bind_method(method(f"    open('b.py','w').write({candidate!r})\n"
        "    return {'action':'evaluate','arguments':{'model':'b.py'}}\n"))
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob('*') if path.is_dir()}
    result = agent.execute_method(lab)
    assert result['status'] == 'method_no_progress'
    assert result['no_progress']['steps'] == [0, 1]
    assert result['no_progress']['model'] == 'b.py'
    assert result['no_progress']['mse'] == 1.
    assert len(result['method_results']) == 2
    assert all(item['feedback']['retention_decision']['status'] == 'incumbent_retained'
               for item in result['method_results'])
    assert result['costs']['compute_cost'] == 3
    assert lab.paid == 1 and result['fresh_confirmation'] is None
    assert result['costs']['model_calls'] == 0
    assert agent._procedure_rejected_context is None
    assert (tmp_path / '_evaluated_model/a.py').read_bytes() == incumbent
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob('*') if path.is_dir()} == before


def test_improved_rejected_candidate_replaces_baseline_before_unchanged_handoff(tmp_path):
    agent, lab, _ = prepared(tmp_path, ExperimentalDesignMethodAgent)
    write(agent, 'a.py', "def predict(row): return 1.\n", lab)
    agent._execute('evaluate', {'model': 'a.py'}, lab)
    write(agent, 'b.py', "def predict(row): return 2.\n", lab)
    agent.bind_method(method(
        "    if step == 1: open('b.py','w').write('def predict(row): return 0.\\n')\n"
        "    return {'action':'evaluate','arguments':{'model':'b.py'}}\n"))
    result = agent.execute_method(lab)
    assert result['status'] == 'method_no_progress'
    assert result['no_progress']['steps'] == [1, 2]
    assert [item['feedback']['mse'] for item in result['method_results']] == [4., 0., 0.]
    assert [item['feedback']['retention_decision']['status']
            for item in result['method_results']] == [
                'incumbent_retained', 'candidate_retained', 'incumbent_retained']
    assert result['evaluated_model']['source_model'] == 'b.py'
    assert lab.paid == 1 and result['fresh_confirmation'] is None


@pytest.mark.parametrize('revision', ['source', 'dependency', 'bound_dependency', 'observations',
                                      'fit_parameters', 'score_improvement'])
def test_rejected_candidate_real_changes_do_not_trigger_handoff(tmp_path, revision):
    agent, lab = rejected_candidate(tmp_path)
    change = ''
    if revision == 'source':
        change = "    if step == 1: open('b.py','w').write(\"def predict(row): return 1. + 2. * row['features']['u']\\n\")\n"
    elif revision == 'dependency':
        change = "    if step == 1: open('scale.txt','w').write('2')\n"
    elif revision == 'bound_dependency':
        write(agent, 'bank/_evaluated_model/child.py', "def predict(row): return row['features']['u']\n", lab)
        write(agent, 'model_bindings.json', json.dumps({'b.py': {'child': 'bank/_evaluated_model/child.py'}}), lab)
        write(agent, 'b.py', "def bind_models(models):\n    global child\n    child = models['child']\ndef predict(row): return 1. + child(row)\n", lab)
        change = "    if step == 1: open('bank/_evaluated_model/child.py','w').write(\"def predict(row): return 2. * row['features']['u']\\n\")\n"
    elif revision == 'observations':
        change = f"    if step == 1: return {{'actions': [{action(0.)!r}]}}\n"
    else:
        write(agent, 'iterations.txt', '0', lab)
        prediction = '1.' if revision == 'fit_parameters' else '1. / iterations'
        write(agent, 'b.py', "def fit(rows):\n    global iterations\n"
            "    iterations = int(open('iterations.txt').read()) + 1\n"
            "    open('iterations.txt','w').write(str(iterations))\n"
            f"def predict(row): return {prediction}\n", lab)
    last_evaluation = 2 if revision == 'observations' else 1
    agent.bind_method(method(change +
        f"    if step > {last_evaluation}: return {{'stop':'Revised candidate evaluated.'}}\n"
        "    return {'action':'evaluate','arguments':{'model':'b.py'}}\n"))
    result = agent.execute_method(lab)
    assert result['status'] == 'method_stopped' and 'no_progress' not in result
    evaluations = [item for item in result['method_results']
                   if (item['proposal'] or {}).get('action') == 'evaluate']
    assert len(evaluations) == 2
    assert [item['feedback']['mse'] for item in evaluations] == (
        [1., .25] if revision == 'score_improvement' else [1., 1.])
    assert all(item['feedback']['retention_decision']['status'] == 'incumbent_retained'
               for item in evaluations)
    assert result['fresh_confirmation'] is None and result['costs']['model_calls'] == 0


# Recorded trial018 paid scores. Replaying these checks controller behavior only;
# it does not rerun the scientific fit or claim a new scientific result.
TRIAL018_SCORES = (
    2.9995952921789594e-28, 2.9996893747151137e-28, 2.9995059300468683e-28,
    2.99894631800656e-28, 3.000343069291576e-28, 2.9991022541131962e-28,
    2.998692881140138e-28, 2.998581870258378e-28, 2.9988838133458626e-28,
    2.9995698996605507e-28, 2.9991016030229806e-28, 2.9985478507946128e-28,
)

# Seven evaluations of the unchanged trial037 candidate. These are the paid-fit
# scores, not its separate grouped-fold validation or independent confirmation.
TRIAL037_SCORES = (
    0.03778975760356935, 0.03778975760356938, 0.03778975760356934,
    0.03778975760356933, 0.037789757603569354, 0.03778975760356937,
    0.037789757603569334,
)


def recorded_score_replay(tmp_path, monkeypatch, scores, *, threshold=1e-6, perfect_incumbent=False):
    agent, lab, _ = prepared(tmp_path, ExperimentalDesignMethodAgent)
    agent.config = replace(agent.config, confirmation_mse_threshold=threshold)
    agent._execute('query', {'actions': [action(8.)]}, lab)
    write(agent, 'a.py', "def predict(row): return row['features']['u']\n"
          if perfect_incumbent else 'def predict(row): return 0.\n', lab)
    agent._execute('evaluate', {'model': 'a.py'}, lab)
    candidate = 'shared/expanded_history_fit/_evaluated_model/model.py'
    write(agent, candidate, "def predict(row): return row['features']['u']\n", lab)
    actual_score = agent._score
    recorded = iter(scores)
    consumed = []

    def score(model):
        if model != candidate:
            return actual_score(model)
        value = next(recorded)
        consumed.append(value)
        agent.costs['compute_cost'] += len(agent.rows)
        return {'returncode': 0, 'mse': value, 'rows': len(agent.rows)}

    monkeypatch.setattr(agent, '_score', score)
    agent.bind_method(method(
        "    if step > 1: return {'stop':'Compared the completed scores.'}\n"
        f"    return {{'action':'evaluate','arguments':{{'model':{candidate!r}}}}}\n"))
    return agent, lab, consumed


@pytest.mark.parametrize('scores', [TRIAL018_SCORES, TRIAL018_SCORES[1:]])
def test_nested_retained_candidate_roundoff_returns_actual_scores_after_two_evaluations(tmp_path, monkeypatch, scores):
    agent, lab, consumed = recorded_score_replay(tmp_path, monkeypatch, scores)
    result = agent.execute_method(lab)
    assert result['status'] == 'method_no_progress'
    assert result['no_progress']['steps'] == [0, 1]
    assert consumed == list(scores[:2])
    assert [item['feedback']['mse'] for item in result['method_results']] == consumed
    assert result['no_progress']['previous_mse'] == consumed[0]
    assert result['no_progress']['mse_roundoff_tolerance'] == math.ulp(8.) ** 2
    assert [item['feedback']['retention_decision']['status']
            for item in result['method_results']] == [
                'candidate_retained',
                'candidate_retained' if consumed[1] < consumed[0] else 'incumbent_retained']
    assert result['evaluated_model']['mse'] == min(consumed)
    assert result['costs']['compute_cost'] == 6
    assert lab.paid == 2 and result['fresh_confirmation'] is None
    assert result['costs']['model_calls'] == 0


@pytest.mark.parametrize('scores,threshold', [
    (TRIAL018_SCORES[:2], 1e-30),  # Similar rounding above the actual confirmation gate.
    ((3e-28, 1e-28), 1e-6),       # Clear improvement even though both are below the gate.
])
def test_changed_scores_not_hidden_by_roundoff_handoff(tmp_path, monkeypatch, scores, threshold):
    agent, lab, consumed = recorded_score_replay(tmp_path, monkeypatch, scores, threshold=threshold)
    result = agent.execute_method(lab)
    assert result['status'] == 'method_stopped' and 'no_progress' not in result
    assert consumed == list(scores)
    assert result['evaluated_model']['mse'] == min(scores)
    assert lab.paid == 2 and result['fresh_confirmation'] is None
    assert result['costs']['model_calls'] == 0


@pytest.mark.parametrize('scores', [TRIAL037_SCORES, TRIAL037_SCORES[1:]])
@pytest.mark.parametrize('perfect_incumbent', [False, True])
def test_nonzero_paid_score_roundoff_returns_to_author_without_rescoring(tmp_path, monkeypatch, scores, perfect_incumbent):
    agent, lab, consumed = recorded_score_replay(
        tmp_path, monkeypatch, scores, perfect_incumbent=perfect_incumbent)
    result = agent.execute_method(lab)
    assert result['status'] == 'method_no_progress'
    assert result['no_progress']['steps'] == [0, 1]
    assert consumed == list(scores[:2])
    assert [entry['feedback']['mse'] for entry in result['method_results']] == consumed
    assert result['no_progress']['previous_mse'] == consumed[0]
    assert result['no_progress']['mse_roundoff_tolerance'] == 8 * math.ulp(max(consumed))
    assert result['evaluated_model']['mse'] == (0. if perfect_incumbent else min(consumed))
    assert [entry['feedback']['retention_decision']['status']
            for entry in result['method_results']] == (['incumbent_retained'] * 2 if perfect_incumbent else [
                'candidate_retained',
                'candidate_retained' if consumed[1] < consumed[0] else 'incumbent_retained'])
    assert result['costs']['compute_cost'] == 6
    assert lab.paid == 2 and result['fresh_confirmation'] is None
    assert result['costs']['model_calls'] == 0


@pytest.mark.parametrize('scores', [
    (TRIAL037_SCORES[0], TRIAL037_SCORES[0] - 1e-10),
    (math.nextafter(1e-6, math.inf), 1e-6),
    (1e-6, math.nextafter(1e-6, math.inf)),
])
def test_nonzero_improvement_or_confirmation_gate_transition_is_not_roundoff(tmp_path, monkeypatch, scores):
    agent, lab, consumed = recorded_score_replay(tmp_path, monkeypatch, scores)
    result = agent.execute_method(lab)
    assert result['status'] == 'method_stopped' and 'no_progress' not in result
    assert consumed == list(scores)
    assert result['evaluated_model']['mse'] == min(scores)
    assert lab.paid == 2 and result['fresh_confirmation'] is None
    assert result['costs']['model_calls'] == 0
