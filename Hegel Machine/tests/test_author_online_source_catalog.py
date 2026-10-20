"""Author source discovery must not expand the frozen practice archive."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.anonymous_method_learning import AnonymousResearchMethodLearner
from hegel_machine.cognition.code_agent import retain_evaluated_python_model
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_anonymous_method_learning import laboratory, trial_preparer
from scripts.run_anonymous_method_revision import attach_online_research_experience


PROGRAM = '''def fit(rows):
    global offset
    offset = sum(row['target'] for row in rows) / len(rows)
def predict(row):
    return offset
'''
METHOD = "def select_actions(spec, records, memory): return {'stop': 'fixture'}\n"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_confirmed_program_joins_author_catalog_once_without_entering_practice(tmp_path):
    source = tmp_path / 'online'
    write_json(source / 'model_observations.json', [{'features': {'u': 0.}, 'target': 3.}])
    (source / 'model.py').write_text(PROGRAM)
    submitted_directory = Path('_submitted_models/confirmation_0000/_evaluated_model')
    retained = retain_evaluated_python_model(source, 'model.py', [], 0.,
                                            destination=submitted_directory)
    submitted = {'model': str(submitted_directory / retained['model']),
                 'source_model': 'model.py', 'paid_rows': 1, 'mse': 0.}
    write_json(source / 'training.json', {'status': 'completed', 'model': 'model.py',
        'costs': {'observation_cost': 97}, 'events': [],
        'fresh_confirmation': {'passed': True, 'mse': 0., 'rows': 96,
                               'submitted_candidate': submitted}})
    write_json(source / 'evaluation.json', {'terminal_marker': 'not author experience'})
    # Only the submitted execution supports the catalogue entry, not later edits.
    (source / 'model.py').write_text('def predict(row): return -999.\n')
    original_source_files = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}

    old = {'id': 'source_000', 'directory': 'archive/old',
           'model': 'archive/old/model.py', 'paid_trace': 'archive/old/paid_trace.json',
           'status': 'completed'}
    original = {'files': {METHOD_PATH: METHOD,
        'archive/old/model.py': 'def predict(row): return 0.\n',
        'source_catalog.json': json.dumps([old])},
        'source_catalog': [old], 'research_roles': {'predictors': [old['model']]}}
    before = deepcopy(original)
    case = {'kind': 'memoryless', 'seed': 113, 'prefix_seed': 317, 'confirmation_seed': 419}
    limits = InquiryLLMConfig(observation_budget=192, compute_budget=1000,
                             execution_seconds=30., confirmation_rows=96)
    prepare = trial_preparer(original, [], tmp_path / 'reference_archive')
    author = AnonymousResearchMethodLearner(None, tmp_path / 'author', knowledge=original,
        cases={'case_000': case}, lab_factory=laboratory, prepare_trial=prepare,
        config=limits, practice_limits=limits)

    first = attach_online_research_experience(author, source)
    first_catalog = json.loads(author.files['source_catalog.json'])
    second = attach_online_research_experience(author, source)
    catalog = json.loads(author.files['source_catalog.json'])
    assert catalog == first_catalog and len(catalog) == 2
    assert catalog[0] == old
    entry = catalog[1]
    assert entry['id'] == first['source_catalog_entry'] == second['source_catalog_entry']
    assert entry['entry_points'] == ['fit', 'predict']
    assert 'author experience' in entry['scope']
    assert 'coefficients' in entry['program_reuse']
    assert (author.workspace / entry['model']).read_text() == PROGRAM
    assert author.workspace / entry['directory'] == (author.workspace / entry['model']).parent
    assert json.loads((author.workspace / entry['paid_trace']).read_text())['status'] == 'completed'
    request = author._request(author.interface)
    assert request['session_knowledge']['source_catalog'] == catalog
    assert request['session_knowledge']['files']['source_catalog.json'] == author.files['source_catalog.json']
    assert json.loads((author.workspace / 'source_catalog.json').read_text()) == catalog
    assert set(author.knowledge['files']) == set(before['files'])
    assert {key: value for key, value in author.knowledge['files'].items()
            if key != 'source_catalog.json'} == {
        key: value for key, value in before['files'].items() if key != 'source_catalog.json'}

    practice = prepare(case, tmp_path / 'practice', METHOD, limits)['agent']
    assert original == before
    assert practice.knowledge == before
    assert json.loads((practice.workspace / 'source_catalog.json').read_text()) == [old]
    assert not (practice.workspace / 'online_research_experience').exists()
    assert all('online_confirmed_source' not in text for text in practice.files.values())
    assert not (author.workspace / 'online_research_experience/evaluation.json').exists()
    assert all(path.read_bytes() == contents for path, contents in original_source_files.items())
