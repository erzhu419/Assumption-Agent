"""Store duplicate inherited source text once without changing its versions."""

from copy import deepcopy
import json

from hegel_machine.cognition.code_provider_context import load_code_context, prepare_code_context


def test_stored_request_references_exact_duplicates_and_retains_changed_versions(tmp_path):
    trace = 'archive/source_000/paid_trace.json'
    source = 'large old evidence ' * 10000
    program = 'def predict(row): return 3.\n' + '# retained source\n' * 1000
    request = {'files': {trace: source, 'old_program.py': program,
                        'model.py': 'new predictor', 'new.py': 'new file'},
        'session_knowledge': {'files': {trace: source, 'old_program.py': program,
                                      'model.py': 'old predictor'}}}
    original = deepcopy(request)
    context = prepare_code_context(tmp_path, request, schema={}, instructions='fixture')
    saved = json.loads((tmp_path / 'research_context.json').read_text())
    assert saved['files'][trace] == {'full_content_file': 'research_context.json',
        'json_path': ['session_knowledge', 'files', trace]}
    assert saved['files']['model.py'] == 'new predictor'
    assert saved['session_knowledge']['files']['model.py'] == 'old predictor'
    assert load_code_context(tmp_path) == original == request
    assert context['files']['model.py'] == 'new predictor'
    reference = context['files']['old_program.py']
    retrieved = saved
    for key in reference['json_path']:
        retrieved = retrieved[key]
    assert retrieved == program and isinstance(retrieved, str)
    assert (tmp_path / 'research_context.json').stat().st_size < len(json.dumps(original)) - len(source) + 200
    # Recovery reads the saved versions, even after the ordinary working file changes.
    (tmp_path / 'model.py').write_text('later predictor')
    assert load_code_context(tmp_path) == original


def test_original_inline_context_still_loads_without_changes(tmp_path):
    request = {'files': {'model.py': 'new'}, 'session_knowledge': {'files': {'model.py': 'old'}}}
    (tmp_path / 'research_context.json').write_text(json.dumps(request))
    assert load_code_context(tmp_path) == request


def test_online_restore_expands_repeated_source_without_buying_observations(tmp_path):
    from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
    from test_paid_scene_method_repair import make_repair
    from test_temporal_python_refit import LawLab

    name, source = 'shared/old_program.py', '# retained source\n' * 1000
    _, checkpoint, previous = make_repair(tmp_path / 'original', extra_files={name: source})
    request = json.loads((checkpoint.parent / 'research_context.json').read_text())
    prepare_code_context(checkpoint.parent, request, schema={}, instructions='fixture')
    restored = CommonArchiveInquiryAgent(None, tmp_path / 'restored', arm='hegel',
        source_task=False, knowledge={'files': {}}, config=InquiryLLMConfig(**previous['limits']))
    restored.restore_research(previous, checkpoint.parent, LawLab(slope=-2., offset=.4),
                              confirmation_seed=173)
    assert restored.knowledge == request['session_knowledge']
    assert restored.files[name] == source
    assert restored.costs == previous['costs']
