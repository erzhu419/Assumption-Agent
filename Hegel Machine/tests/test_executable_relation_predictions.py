"""Local numerical evidence reaches an authored fit and its actual predictions."""

import json

import pytest

from test_executable_evidence_binding import setup, return_local_checks


RELATION_MODEL = '''SCALE = 1.
def bind_research(inputs):
    global relation
    relation = inputs['contrast']['checks'][0]
def fit(rows):
    global slope
    assert rows and relation['status'] in ('satisfied', 'violated')
    public_contrast = sum(w['coefficient'] * w['features']['u'] for w in relation['witnesses'])
    slope = relation['value'] / public_contrast
def predict(row):
    assert 'target' not in row
    return SCALE * slope * row['features']['u']
'''


def bind_relation(agent, lab):
    record = return_local_checks(agent, lab)
    agent._execute('write', {'path': 'relation.py', 'code': RELATION_MODEL}, lab)
    agent._execute('run', {'operator': 'bind_research', 'model': 'relation.py',
                          'inputs': {'contrast': {'revision_index': record['index']}}}, lab)
    return record


def relation_check(result):
    bound = result['research_bindings'][0]['local_relations'][0]
    assert bound['slot'] == 'contrast'
    return bound['checks'][0]


def test_refuted_local_fact_changes_fit_before_prediction_without_parameter_getter(tmp_path):
    agent, lab = setup(tmp_path)
    record = bind_relation(agent, lab)
    result = agent._execute('evaluate', {'model': 'relation.py'}, lab)
    assert result['returncode'] == 0 and result['mse'] == 0.
    assert 'get_parameters' not in RELATION_MODEL
    check = relation_check(result)
    assert check['measured_value'] == check['predicted_value'] == 2.
    assert check['residual'] == 0.
    assert check['evidence_status'] == 'violated'  # Original expectation was zero.
    assert check['scope'] == record['check_results'][0]['scope']
    assert result['research_prediction_count'] == 2


def test_local_prediction_mismatch_is_reported_without_veto_and_works_during_refit(tmp_path):
    agent, lab = setup(tmp_path)
    bind_relation(agent, lab)
    agent._execute('write', {'path': 'relation.py', 'code':
                           RELATION_MODEL.replace('SCALE = 1.', 'SCALE = 0.')}, lab)
    # Bound witnesses are the first trajectory; the second is held out here.
    result = agent._execute('run', {'operator': 'refit', 'model': 'relation.py',
                                  'validation_records': [1]}, lab)['validation']
    assert result['returncode'] == 0
    assert result['mse'] == pytest.approx(.25)
    check = relation_check(result)
    assert check['measured_value'] == 2. and check['predicted_value'] == 0.
    assert check['residual'] == -2.
    assert check['evidence_status'] == 'violated'
    assert result['research_prediction_count'] == 2


def test_reanalysis_updates_mutable_fit_without_changing_better_frozen_predictor(tmp_path):
    agent, lab = setup(tmp_path)
    record = bind_relation(agent, lab)
    first = agent._execute('evaluate', {'model': 'relation.py'}, lab)
    old_model = first['evaluated_model']['model']
    frozen_input_file = tmp_path / '_evaluated_model/model_research_inputs.json'
    frozen_inputs = frozen_input_file.read_text()
    assert json.loads(frozen_inputs)['relation.py']['contrast']['version'] == 1

    return_local_checks(agent, lab, expected=2., revision_index=record['index'])
    agent._execute('write', {'path': 'relation.py', 'code':
                           RELATION_MODEL.replace('SCALE = 1.', 'SCALE = 0.')}, lab)
    newer = agent._execute('evaluate', {'model': 'relation.py'}, lab)
    assert newer['returncode'] == 0 and newer['mse'] > first['mse']
    assert newer['research_input_update']['inputs']['contrast']['version'] == 2
    assert relation_check(newer)['evidence_status'] == 'satisfied'
    assert relation_check(newer)['residual'] == -2.
    assert frozen_input_file.read_text() == frozen_inputs

    old = agent._execute('evaluate', {'model': old_model}, lab)
    assert old['returncode'] == 0 and old['mse'] == 0.
    assert old['research_bindings'][0]['local_relations'][0]['version'] == 1
    assert relation_check(old)['evidence_status'] == 'violated'
    assert relation_check(old)['residual'] == 0.


@pytest.mark.parametrize('model_kind', ['current', 'frozen', 'dependent'])
def test_refit_reports_actual_bound_validation_labels_before_fitting(tmp_path, model_kind):
    agent, lab = setup(tmp_path)
    bind_relation(agent, lab)
    model = 'relation.py'
    if model_kind == 'frozen':
        model = agent._execute('evaluate', {'model': model}, lab)['evaluated_model']['model']
    elif model_kind == 'dependent':
        model = 'composition.py'
        agent._execute('write', {'path': model, 'code': '''def bind_models(models):
    global child
    child = models['child']
def fit(rows): pass
def predict(row): return child(row)
'''}, lab)
        agent._save('model_bindings.json', json.dumps({model: {'child': 'relation.py'}}))
    compute_before = agent.costs['compute_cost']
    with pytest.raises(ValueError, match='validation labels are already bound into fitting'):
        agent._execute('run', {'operator': 'refit', 'model': model, 'path': 'invalid_split',
                              'validation_records': [0]}, lab)
    assert agent.costs['compute_cost'] == compute_before
    assert not (tmp_path / 'invalid_split').exists()


def test_refit_does_not_treat_an_old_tasks_row_number_as_current_validation_label(tmp_path):
    agent, lab = setup(tmp_path)
    bind_relation(agent, lab)
    model = agent._execute('evaluate', {'model': 'relation.py'}, lab)['evaluated_model']['model']
    frozen = tmp_path / '_evaluated_model/model_research_inputs.json'
    prior = json.loads(frozen.read_text())
    # A distinct source task may use exactly the same response row numbers.
    for witness in prior['relation.py']['contrast']['checks'][0]['witnesses']:
        witness['features']['u'] += 10.
        witness['target'] += 10.
    frozen.write_text(json.dumps(prior))
    result = agent._execute('run', {'operator': 'refit', 'model': model,
                                  'validation_records': [0]}, lab)
    assert result['validation']['returncode'] == 0
    assert result['validation']['mse'] == 0.
