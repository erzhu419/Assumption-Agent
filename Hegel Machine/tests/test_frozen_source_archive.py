"""Freeze concrete toy source outputs without repeating the common archive."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.anonymous_research_archive import prepare_frozen_source_archive
from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.models import Observation


NATIVE = '_experiment_models/native_0000/_evaluated_model/native.py'
MODEL = '''import json
from pathlib import Path
def predict(row): return json.loads(Path('parameter.json').read_text())['gain'] * row['features']['u']
'''
FIT = '''from pathlib import Path
value = 0.
def fit(rows):
    global value
    value = sum(row['target'] for row in rows) / len(rows)
    Path('fit_created.txt').write_text('local fit')
def predict(row): return value
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def source_study(tmp_path):
    root = tmp_path / 'source'
    paths = [f'archive/old_{i:03d}/_evaluated_model/model.py' for i in range(49)]
    catalog = [{'id': f'old_{i}', 'model': path, 'status': 'unconfirmed'}
               for i, path in enumerate(paths)]
    base = {'files': {path: "def predict(row): return row['features']['u']\n" for path in paths},
        'source_catalog': catalog, 'research_roles': {'predictors': paths[:]},
        'lessons': ['Keep the original failed source.'],
        'theory_consolidation': {'file': '/old/fixed_library.json', 'json_path': []}}
    base['files']['archive/old_000/paid_trace.json'] = '{"events":["large unchanged fixture"]}'
    base['files']['source_catalog.json'] = json.dumps(catalog)
    plan = {'cases': [{'id': 'case_0'}, {'id': 'case_1'}], 'arms': ['archive', 'fixed_pr2', 'pr3']}
    write(root / 'protocol.json', plan)
    for case in plan['cases']:
        for arm in plan['arms']:
            scene = root / case['id'] / arm
            for name, content in base['files'].items():
                write(scene / name, content)
            write(scene / NATIVE, MODEL)
            write((scene / NATIVE).parent / 'parameter.json', {'gain': 3.})
            write((scene / NATIVE).parent / 'model_observations.json', [{'target': 'not-needed-by-native'}])
            write(scene / 'fitted.py', FIT)
            write(scene / 'model_observations.json', [{'features': {'u': 0.}, 'history': [], 'target': 7.}])
            write(scene / 'laboratory_observations.json', [{'rows': 'global source facts'}])
            write(scene / 'bad.py', 'def predict(:\n')
            state = {'exported_candidate_paths': [NATIVE, 'fitted.py', 'bad.py'],
                'research_candidates': {}, 'questions': [{'native_condition_research': [{
                    'candidates': [{'model': NATIVE}], 'exported_models': [{'model': NATIVE}]}]}]}
            write(scene / 'research_state.json', state)
            write(scene / 'training.json', {'workspace': str(scene),
                'status': 'method_error' if arm == 'pr3' else 'completed',
                'model': paths[0], 'costs': {'compute_cost': 17},
                'fresh_confirmation': {'passed': False}, 'evaluated_model': None})
    index = root / 'source_aggregation/shared/theory_index.json'
    write(index, {'theory_index': {}})
    write(root / 'source_aggregation.json', {'index_path': str(index)})
    return root, base


def test_shared_archive_keeps_49_paths_all_arms_exports_and_failed_source_refs(tmp_path):
    root, base = source_study(tmp_path)
    original = deepcopy(base)
    shared = prepare_frozen_source_archive(root, base)
    assert base == original
    for name, content in base['files'].items():
        if name != 'source_catalog.json':
            assert shared['files'][name] == content
    assert shared['research_roles']['predictors'][:49] == base['research_roles']['predictors']
    assert len(shared['source_catalog']) == 49 + 6
    assert len(shared['research_roles']['predictors']) == 49 + 18
    for case in ('case_0', 'case_1'):
        for arm in ('archive', 'fixed_pr2', 'pr3'):
            prefix = f'source_research/{case}/{arm}'
            assert shared['files'][f'{prefix}/{NATIVE}'] == MODEL
            assert shared['files'][f'{prefix}/_evaluated_model/bad.py'] == 'def predict(:\n'
            entry = next(e for e in shared['source_catalog'] if e['id'] == f'source_research:{case}:{arm}')
            assert entry['model'] == base['research_roles']['predictors'][0]
            assert entry['source_ref']['file'] == str(root / case / arm / 'training.json')
            assert entry['costs_ref']['json_path'] == ['costs']
            assert entry['observations_ref']['file'] == str(root / case / arm / 'laboratory_observations.json')
            if arm == 'pr3':
                assert entry['status'] == 'method_error'
    additions = {name: value for name, value in shared['files'].items() if name not in base['files']}
    assert not any(name.endswith('paid_trace.json') for name in additions)
    assert not any(name.endswith('laboratory_observations.json') for name in additions)
    assert not any('/archive/old_' in name for name in additions)
    assert not any('not-needed-by-native' in value for value in additions.values())
    assert shared['theory_index'] == {'file': str(root / 'source_aggregation/shared/theory_index.json'), 'json_path': []}
    assert shared['theory_consolidation'] == base['theory_consolidation']


def test_parameters_and_required_fit_view_execute_in_independent_target_contexts(tmp_path):
    root, base = source_study(tmp_path)
    shared = prepare_frozen_source_archive(root, base)
    targets = [tmp_path / 'target_a', tmp_path / 'target_b']
    for target in targets:
        for name, content in shared['files'].items():
            write(target / name, content)
    model = f'source_research/case_0/archive/{NATIVE}'
    fit_model = 'source_research/case_0/archive/_evaluated_model/fitted.py'
    row = [Observation({'u': 2.}, 6., ())]
    for target in targets:
        result = evaluate_python_model(target, model, row)
        assert result['returncode'] == 0 and result['predictions'] == [6.]
        fitted = evaluate_python_model(target, fit_model, [Observation({'u': 0.}, 7., ())])
        assert fitted['returncode'] == 0 and fitted['predictions'] == [7.]
    assert not (root / 'case_0/archive/fit_created.txt').exists()
    write((targets[0] / model).parent / 'parameter.json', {'gain': 99.})
    assert json.loads(((targets[1] / model).parent / 'parameter.json').read_text()) == {'gain': 3.}
    assert json.loads(((root / 'case_0/archive' / NATIVE).parent / 'parameter.json').read_text()) == {'gain': 3.}


def test_initial_archive_with_only_one_top_role_expands_nested_catalog_predictors(tmp_path):
    root, base = source_study(tmp_path)
    first, *remaining = base['research_roles']['predictors']
    base['research_roles']['predictors'] = [first]
    base['source_catalog'] = [{'id': 'top', 'model': first}]
    base['files']['source_catalog.json'] = json.dumps(base['source_catalog'])
    directory = 'archive/retained/_evaluated_model'
    nested = []
    expected = [first]
    for index, old in enumerate(remaining):
        mapped = f'{directory}/{old}'
        base['files'][mapped] = base['files'].pop(old)
        nested.append({'id': f'local_{index}', 'model': old, 'status': 'unconfirmed'})
        expected.append(mapped)
    base['files'][f'{directory}/source_catalog.json'] = json.dumps(nested)
    assert 'finite_native_author' not in base
    shared = prepare_frozen_source_archive(root, base)
    assert shared['research_roles']['predictors'][:49] == expected
    assert len(shared['research_roles']['predictors']) == 49 + 18
    assert all(model in shared['files'] for model in expected)
    assert shared['files'][f'{directory}/source_catalog.json'] == base['files'][f'{directory}/source_catalog.json']


def test_declared_bound_child_context_keeps_its_parameters_at_the_original_relative_address(tmp_path):
    root, base = source_study(tmp_path)
    scene = root / 'case_0/pr3'
    parent = '''models = {}
def bind_models(value):
    global models
    models = value
def predict(row): return 2. * models['child'](row)
'''
    child = 'children/_evaluated_model/child.py'
    write(scene / 'parent.py', parent)
    write(scene / child, MODEL)
    write((scene / child).parent / 'parameter.json', {'gain': 5.})
    write(scene / 'model_bindings.json', {'parent.py': {'child': child}})
    state = json.loads((scene / 'research_state.json').read_text())
    state['exported_candidate_paths'].append('parent.py')
    retained = tmp_path / 'retained_context'
    write(retained / 'model.py', MODEL)
    write(retained / 'parameter.json', {'gain': 3.})
    submitted = scene / '_submitted_models/prior/_evaluated_model'
    submitted.parent.mkdir(parents=True, exist_ok=True)
    submitted.symlink_to(retained, target_is_directory=True)
    state['exported_candidate_paths'].append('_submitted_models/prior/_evaluated_model/model.py')
    write(scene / 'research_state.json', state)
    shared = prepare_frozen_source_archive(root, base)
    target = tmp_path / 'target'
    for name, content in shared['files'].items():
        write(target / name, content)
    prefix = 'source_research/case_0/pr3/_evaluated_model'
    assert shared['files'][f'{prefix}/{child}'] == MODEL
    assert json.loads(shared['files'][f'{prefix}/model_bindings.json'])['parent.py']['child'] == child
    result = evaluate_python_model(target, f'{prefix}/parent.py', [Observation({'u': 2.}, 20., ())])
    assert result['returncode'] == 0 and result['predictions'] == [20.]
    restored = 'source_research/case_0/pr3/_submitted_models/prior/_evaluated_model/model.py'
    result = evaluate_python_model(target, restored, [Observation({'u': 2.}, 6., ())])
    assert result['returncode'] == 0 and result['predictions'] == [6.]
    assert not (target / restored).is_symlink()


def test_continued_source_uses_latest_result_and_original_retained_workspace(tmp_path):
    from shutil import copytree

    original, base = source_study(tmp_path)
    continued = tmp_path / 'continued'
    plan = json.loads((original / 'protocol.json').read_text())
    write(continued / 'protocol.json', plan)
    write(continued / 'source_aggregation.json', json.loads((original / 'source_aggregation.json').read_text()))
    latest = continued / 'case_0/pr3'
    copytree(original / 'case_0/pr3', latest)
    write((latest / NATIVE).parent / 'parameter.json', {'gain': 11.})
    result = json.loads((latest / 'training.json').read_text())
    result.update(workspace=str(latest), status='latest_failed_source')
    write(latest / 'training.json', result)
    trials = [{'case': case['id'], 'arm': arm,
        'training_ref': {'file': str(original / case['id'] / arm / 'training.json'), 'json_path': []},
        **({'result_ref': {'file': str(latest / 'training.json'), 'json_path': []}}
           if (case['id'], arm) == ('case_0', 'pr3') else {})}
        for case in plan['cases'] for arm in plan['arms']]
    write(continued / 'summary.json', {'trials': trials})
    assert not (continued / 'case_0/archive').exists()
    shared = prepare_frozen_source_archive(continued, base)
    selected = {entry['id']: entry for entry in shared['source_catalog'] if entry['id'].startswith('source_research:')}
    assert selected['source_research:case_0:pr3']['source_ref'] == trials[2]['result_ref']
    assert selected['source_research:case_0:pr3']['status'] == 'latest_failed_source'
    assert selected['source_research:case_0:archive']['source_ref'] == trials[0]['training_ref']
    prefix = 'source_research/case_0/pr3/'
    assert json.loads(shared['files'][prefix + str(Path(NATIVE).parent / 'parameter.json')]) == {'gain': 11.}


def test_source_cohort_preserves_existing_source_research_bodies(tmp_path):
    root, base = source_study(tmp_path)
    old_model = f'source_research/case_0/pr3/{NATIVE}'
    old_body = 'def predict(row): return -731.\n'
    base['files'][old_model] = old_body
    # A previously frozen cohort can also already occupy the preferred root.
    prior_cohort_model = str(Path('source_research') / root.parent.name / root.name / 'retained.py')
    base['files'][prior_cohort_model] = 'def predict(row): return -732.\n'
    base['research_roles']['predictors'].extend([old_model, prior_cohort_model])
    old_entry = {'id': 'source_research:case_0:pr3', 'model': old_model,
        'predictors': [old_model], 'status': 'prior_failed_source'}
    base['source_catalog'].append(old_entry)
    base['files']['source_catalog.json'] = json.dumps(base['source_catalog'])
    original = deepcopy(base)
    shared = prepare_frozen_source_archive(root, base)
    assert base == original
    assert shared['files'][old_model] == old_body
    assert shared['files'][prior_cohort_model] == original['files'][prior_cohort_model]
    assert all(entry in shared['source_catalog'] for entry in original['source_catalog'])
    ids = [entry['id'] for entry in shared['source_catalog']]
    assert len(ids) == len(set(ids))
    current = next(entry for entry in shared['source_catalog']
                   if entry.get('source_ref', {}).get('file') == str(root / 'case_0/pr3/training.json'))
    assert current['id'] != old_entry['id']
    exported = next(item['model'] for item in current['model_sources'] if item['source_model'] == NATIVE)
    assert exported != old_model
    assert shared['files'][exported] == MODEL
    assert {old_model, prior_cohort_model, exported}.issubset(shared['research_roles']['predictors'])
