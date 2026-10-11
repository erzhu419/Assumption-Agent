"""Keep complete prior research readable without repeating its data in each prompt."""

import json

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
    assert context['files'][name] == {'full_content_file': name, 'characters': len(source)}
    assert context['files'][trace]['full_content_file'] == trace
    assert context['files']['shared/consolidated_method.py'] == files['shared/consolidated_method.py']
    assert json.loads((tmp_path / 'research_context.json').read_text()) == request
    assert request['files'][name] == source
