"""Keep complete prior research readable without repeating its data in each prompt."""

import json
from copy import deepcopy

from hegel_machine.cognition.code_provider_context import prepare_code_context


def test_catalogued_archives_remain_exactly_retrievable(tmp_path):
    name = 'archive/source_000/_evaluated_model/model.py'
    trace = 'archive/source_000/paid_trace.json'
    source = 'def predict(row):\n    return 7.\n'
    files = {name: source, trace: '{"events": []}',
             'shared/consolidated_method.py': 'def select_actions(*args): return {}\n'}
    request = {'files': files, 'session_knowledge': {'files': dict(files),
        'source_catalog': [{'directory': 'archive/source_000/_evaluated_model',
                            'model': name, 'paid_trace': trace}]}}
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    assert name not in context['files'] and trace not in context['files']
    entry = context['session_knowledge']['source_catalog'][0]
    assert entry['model'] == name and entry['paid_trace'] == trace
    assert entry['context_file_count'] == 2
    assert context['session_knowledge']['shared_files'] == ['shared/consolidated_method.py']
    assert context['files']['shared/consolidated_method.py'] == files['shared/consolidated_method.py']
    assert json.loads((tmp_path / 'research_context.json').read_text()) == request
    assert request['files'][name] == source


def test_nested_archive_files_stay_on_disk_and_in_full_request_without_prompt_enumeration(tmp_path):
    files, catalog = {}, []
    for source in range(2):
        directory = f'archive/source_{source:03d}/_evaluated_model'
        model = f'{directory}/model.py'
        trace = f'archive/source_{source:03d}/paid_trace.json'
        catalog.append({'directory': directory, 'model': model, 'paid_trace': trace,
                        'status': 'completed' if source == 0 else 'token_budget_exhausted'})
        files[model] = f'def predict(row): return {source}.\n'
        files[trace] = json.dumps({'events': [{'action': 'query', 'evidence': [source]}]})
        for index in range(160):
            name = f'{directory}/saved_{index:03d}/_evaluated_model/analysis_{index:03d}.py'
            files[name] = f'print("paid evidence {source}, {index}")\n'
    current = {'model.py': 'def predict(row): return 3.\n',
               'shared/consolidated_method.py': 'def select_actions(*args): return {}\n',
               'archive/local_notes.py': 'print("current uncatalogued work")\n'}
    files.update(current)
    inherited = dict(files)
    inherited['archive/source_001/_evaluated_model/inherited_only.py'] = 'print("older analysis")\n'
    inherited['shared/consolidated_method.py'] = 'def select_actions(*args): return {"stop": "old"}\n'
    for name, contents in {**inherited, **files}.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    request = {'files': files, 'session_knowledge': {'files': inherited, 'source_catalog': catalog}}
    original = deepcopy(request)

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    assert context['files'] == current
    assert context['session_knowledge']['shared_files'] == list(current)
    assert context['session_knowledge']['files'] == {
        'shared/consolidated_method.py': inherited['shared/consolidated_method.py']}
    assert context['session_knowledge']['source_catalog'] == [
        {**catalog[0], 'context_file_count': 162}, {**catalog[1], 'context_file_count': 163}]
    prompt = json.dumps(context)
    assert 'analysis_000.py' not in prompt and 'analysis_159.py' not in prompt
    assert 'inherited_only.py' not in prompt
    assert len(prompt) < 5000
    for entry in catalog:
        assert entry['directory'] in prompt and entry['model'] in prompt and entry['paid_trace'] in prompt
    assert request == original
    assert json.loads((tmp_path / 'research_context.json').read_text()) == original
    for name, contents in {**inherited, **files}.items():
        assert (tmp_path / name).read_text() == contents


def test_disk_only_source_catalog_keeps_navigation_without_listing_its_files(tmp_path):
    entry = {'directory': 'repair_archive/source_000',
             'model': 'repair_archive/source_000/model.py',
             'paid_trace': 'repair_archive/source_000/paid_trace.json'}
    files = {'model.py': 'def predict(row): return 2.\n'}
    for name, contents in ((entry['model'], 'def predict(row): return 7.\n'),
                           (entry['paid_trace'], '{"events": []}')):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    request = {'files': files, 'session_knowledge': {'source_catalog': [entry]}}

    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')

    assert context['session_knowledge']['source_catalog'] == [entry]
    assert context['files'] == files
    assert (tmp_path / entry['model']).read_text() == 'def predict(row): return 7.\n'
    assert json.loads((tmp_path / 'research_context.json').read_text()) == request
