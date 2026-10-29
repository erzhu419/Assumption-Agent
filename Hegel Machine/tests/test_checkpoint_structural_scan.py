"""Retained score/program spans stay exact without decoding unrequested history."""

import json
import mmap

import pytest

from hegel_machine.cognition import checkpoint_storage as storage


def _assert_exact_end(value, *, ensure_ascii=True, prefix=b'  ', tail=b',"later":[{},[]]}'):
    encoded = json.dumps(value, ensure_ascii=ensure_ascii, separators=(',', ':')).encode()
    text = prefix + encoded + tail
    _, decoded_end = json.JSONDecoder().raw_decode(encoded.decode())
    expected = len(prefix) + len(encoded.decode()[:decoded_end].encode())
    end = storage._json_value_end(text, len(prefix))
    assert end == expected
    assert json.loads(text[len(prefix):end]) == value


@pytest.mark.parametrize('ensure_ascii', [False, True])
def test_many_retained_paid_scores_and_programs_have_the_exact_json_boundary(ensure_ascii):
    program = {'op': 'add', 'children': [{'op': 'feature', 'name': 'u'},
        {'op': 'const', 'value': -0.4}]}
    source = ('def predict(row):\n    text = "quoted \\\" [{]} \\\\ 原反例"\n'
              '    return row["u"] + 0.4\n')
    value = {'operator_results': [{'model': f'candidate_{index:04d}.py',
        'program': program, 'source': source, 'residuals': [index / 97., -0., None],
        'decision': {'status': 'incumbent_retained', 'costs': {'compute_cost': index}}}
        for index in range(1700)], 'unique_failure': {'stderr': '[{not a container}]'}}
    _assert_exact_end(value, ensure_ascii=ensure_ascii)


@pytest.mark.parametrize('slash_count', [1, 2, 3, 4])
@pytest.mark.parametrize('offset', [-1, 0, 1])
@pytest.mark.parametrize('ending', ['escaped_quote', 'closing_quote'])
def test_quoted_source_and_backslash_runs_cross_scan_chunks(slash_count, offset, ending):
    # The initial objects force the large-container path before its first
    # chunk. Both an escaped quote and a closing quote following an even
    # encoded backslash run occur around each plausible chunk boundary.
    prefix = b'[' + b'{"paid":[]},' * 120
    string_prefix = b'"source '
    first_tokens = list(storage._JSON_CONTAINER_BOUNDARY.finditer(prefix))
    boundary = first_tokens[255].end() + 256 * 1024 + offset
    filler = b'x' * (boundary - len(prefix) - len(string_prefix))
    source = '\\' * slash_count
    if ending == 'escaped_quote':
        source += '" } ] { [ source continues'
    encoded = json.dumps(source).encode()
    text = (prefix + string_prefix + filler + encoded[1:-1]
            + b'",{"after":{"cursor":[1,3],"paid":true}}]')
    _, end = json.JSONDecoder().raw_decode(text.decode())
    assert storage._json_value_end(text + b',"unrelated":[{}]', 0) == end


def test_nested_container_ends_before_its_later_sibling_in_mapped_checkpoint(tmp_path):
    nested = [{'score': {'residuals': [index, -index], 'source': '[{quoted}]'}}
              for index in range(20000)]
    encoded = json.dumps(nested, separators=(',', ':')).encode()
    prefix = b'{"questions":'
    path = tmp_path / 'research_state.json'
    path.write_bytes(prefix + encoded + b',"costs":{"compute_cost":1757873}}\n')
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as text:
        end = storage._json_value_end(text, len(prefix))
        assert end == len(prefix) + len(encoded)
        assert text[end:end + 8] == b',"costs"'


def test_unrequested_large_paid_history_is_located_without_decoding(tmp_path, monkeypatch):
    path = tmp_path / 'research_state.json'
    history = [{'evaluations': [{'program': {'op': 'feature', 'name': 'u'},
        'residuals': [0.1, 0.2], 'costs': {'compute_cost': index}}]}
        for index in range(5000)]
    path.write_text(json.dumps({'questions': [{'id': 'parent'}],
        'operator_results': history, 'costs': {'compute_cost': 1757873}}))
    loads = storage.json.loads

    def selected_values_only(value, *args, **kwargs):
        assert len(value) < 1024, 'unrequested paid score history was decoded'
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(storage.json, 'loads', selected_values_only)
    assert storage.read_json_checkpoint(path, keys={'costs'}) == {
        'costs': {'compute_cost': 1757873}}


def test_completed_selected_root_scan_reuses_positions_for_outgoing_manifest(tmp_path, monkeypatch):
    path = tmp_path / 'training.json'
    value = {'questions': [{'id': 'parent', 'meaning_version': 1}],
        'operator_results': [{'evaluations': list(range(20000))}],
        'outgoing_knowledge': {'manifest': {'hypotheses': [{'path': 'shared/model.py'}]}},
        'costs': {'compute_cost': 1757873}}
    path.write_text(json.dumps(value))
    assert storage.read_json_checkpoint(path, keys={'questions', 'costs', 'absent'}) == {
        'questions': value['questions'], 'costs': value['costs']}

    def already_located(*_args):
        pytest.fail('the complete selected root was scanned again for its manifest')

    monkeypatch.setattr(storage, '_json_value_end', already_located)
    assert storage.read_json_checkpoint(path, keys={'outgoing_knowledge'}) == {
        'outgoing_knowledge': value['outgoing_knowledge']}


def test_early_question_selection_does_not_visit_later_paid_history(tmp_path, monkeypatch):
    questions = [{'id': 'parent', 'cursor': {'binding': 7, 'phase': 'fit'}}]
    prefix = json.dumps({'questions': questions})[:-1]
    path = tmp_path / 'research_state.json'
    path.write_text(prefix + ',"operator_results":[' + '0,' * 100000 + '0]}')
    locate = storage._json_value_end

    def leading_only(text, start):
        assert start < len(prefix), 'question lookup scanned later paid history'
        return locate(text, start)

    monkeypatch.setattr(storage, '_json_value_end', leading_only)
    assert storage.read_json_checkpoint(path, keys={'questions'}) == {'questions': questions}
