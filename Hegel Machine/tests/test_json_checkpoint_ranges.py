"""Moved retained fields share bytes without changing ordinary JSON recovery."""
from copy import deepcopy
from contextlib import ExitStack
import errno
import json
import math
import os
import shutil
import struct
import tracemalloc

import pytest

from hegel_machine.cognition import checkpoint_storage as storage


@pytest.mark.parametrize('ensure_ascii', [True, False])
def test_compact_encoding_uses_bounded_C_batches_with_identical_bytes(
        tmp_path, monkeypatch, ensure_ascii):
    # A new snapshot and a new comparison have no retained range to copy. Do
    # not regress to millions of Python encoder tokens or one giant JSON text.
    value = {'rows': [{'index': i, 'source': 'quoted " []{} \\ 原记录🧪',
                       'residuals': (1.0, -0.0, None, True, False)}
                      for i in range(1800)],
             'source': ('escaped " source \\ 原失败🧪' * 12000),
             'keys': {1: 'integer', None: 'null key'}}
    expected = json.dumps(value, ensure_ascii=ensure_ascii).encode() + b'\n'
    iterencode = json.JSONEncoder.iterencode
    calls = []

    def C_only(encoder, item, _one_shot=False):
        assert _one_shot, 'Compact checkpoint encoding must use bounded C batches'
        calls.append(item)
        chunks = iterencode(encoder, item, _one_shot=_one_shot)
        assert sum(len(chunk.encode('utf-8')) for chunk in chunks) <= storage._JSON_ENCODE_CHUNK
        return chunks

    monkeypatch.setattr(json.JSONEncoder, 'iterencode', C_only)
    path = tmp_path / 'research_state.json'
    storage.write_json_checkpoint(path, value, indent=None, ensure_ascii=ensure_ascii)
    assert path.read_bytes() == expected
    assert len(calls) > 2


def test_new_large_snapshot_has_bounded_encoder_memory(tmp_path):
    value = {'retained_source': 'original counterexample " 原失败🧪' * 300000,
             'rows': [{'index': i, 'loss': i / 97} for i in range(4000)]}
    tracemalloc.start()
    try:
        storage.write_json_checkpoint(tmp_path / 'research_state.json', value, indent=None)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024
    assert json.loads((tmp_path / 'research_state.json').read_bytes()) == value


def test_one_save_reuses_large_source_member_spans_and_equality(tmp_path, monkeypatch):
    path = tmp_path / 'research_state.json'
    value = [{'index': i, 'source': 'old paid failure ' * 10} for i in range(2000)]
    path.write_text(json.dumps({'evaluations': value}))
    compare = storage._JsonSource._value_equal
    comparisons = []

    def counted(source, node, current, allow_nan=True):
        if current is value:
            comparisons.append(node[1])
        return compare(source, node, current, allow_nan)

    monkeypatch.setattr(storage._JsonSource, '_value_equal', counted)
    with ExitStack() as opened:
        source = storage._JsonSource(path, opened)
        source.comparison_encoders = (json.JSONEncoder(),)
        node = source.members()['evaluations']
        first = source.members(node)
        assert source.members(node) is first
        assert source.value_equal(node, value)
        assert source.value_equal(node, value)
        assert comparisons == [node[1]]
        assert source._member_cache_entries <= storage._JSON_MEMBER_CACHE_ENTRIES


def test_writer_reuses_only_its_unchanged_published_root_layout(tmp_path, monkeypatch):
    path = tmp_path / 'research_state.json'
    value = {'history': [{'index': i, 'failure': 'retained counterexample' * 12}
                         for i in range(900)], 'compute': 17}
    storage.write_json_checkpoint(path, value, indent=None)
    storage.write_json_checkpoint(path, value, indent=None)
    members = storage._JsonSource.members
    roots = []

    def published_root(source, node=None, **kwargs):
        if node is None:
            assert source.fields is not None, 'rescanned an unchanged published root'
            roots.append(source.path)
        return members(source, node, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(storage._JsonSource, 'members', published_root)
        value['compute'] = 18
        storage.write_json_checkpoint(path, value, indent=None)
        storage.write_json_checkpoint(path, value, indent=None)
    assert roots and storage.read_json_checkpoint(path) == value
    _, fields = storage._checkpoint_layouts[path.absolute()]
    assert fields['history'][0] is None  # Metadata spans hold no live history.
    value['history'][-1]['failure'] = 'authored change of the same supported file'
    path.write_text(json.dumps(value))
    with ExitStack() as opened:
        source = storage._JsonSource(path, opened)
        assert source.fields is None
    assert storage.read_json_checkpoint(path) == value


def test_published_layout_points_past_alignment_spaces_to_the_actual_json_value(
        tmp_path, monkeypatch):
    shared_filesystem(monkeypatch)
    path = tmp_path / 'research_state.json'
    value = {'header': 'old', 'history': list(range(20000)), 'cost': 17}
    path.write_text(json.dumps(value))
    value['header'] = 'a longer authored header before the unchanged history'
    storage.write_json_checkpoint(path, value, indent=None)
    _, fields = storage._checkpoint_layouts[path.absolute()]
    with path.open('rb') as saved:
        saved.seek(fields['history'][1])
        assert saved.read(1) == b'['
    assert fields['history'][0] is None and fields['cost'][0] == 17
    value['history'][-1] = -1
    value['cost'] = 18
    storage.write_json_checkpoint(path, value, indent=None)
    assert storage.read_json_checkpoint(path) == value


def test_small_published_metadata_does_not_evict_a_large_checkpoint_layout(tmp_path, monkeypatch):
    from collections import OrderedDict
    shared_filesystem(monkeypatch)
    monkeypatch.setattr(storage, '_checkpoint_layouts', OrderedDict())
    path = tmp_path / 'research_state.json'
    value = {'history': list(range(20000)), 'cost': 17}
    path.write_text(json.dumps(value))
    storage.write_json_checkpoint(path, value, indent=None)
    assert path.absolute() in storage._checkpoint_layouts
    for index in range(20):
        metadata = tmp_path / f'metadata_{index}.json'
        storage.write_json_checkpoint(metadata, {'cost': index}, base=path, indent=None)
        assert storage.read_json_checkpoint(metadata) == {'cost': index}
        assert metadata.absolute() not in storage._checkpoint_layouts
    assert path.absolute() in storage._checkpoint_layouts


def test_bounded_encoder_failure_keeps_previous_complete_snapshot(tmp_path):
    path = tmp_path / 'research_state.json'
    original = {'paid': 97, 'retained': 'original failure ' * 10000}
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    cycle = {'source': 'new actual action'}
    cycle['self'] = cycle
    with pytest.raises(ValueError, match='Circular reference'):
        storage.write_json_checkpoint(path, {**original, 'new': cycle}, indent=None)
    assert path.read_bytes() == before
    assert not path.with_name(path.name + '.writing').exists()


def test_large_retained_json_uses_file_backed_bytes_not_a_unicode_copy(tmp_path):
    source = tmp_path / 'research_state.json'
    with source.open('wb') as output:
        output.write(b'{"retained":["')
        for _ in range(1024):
            output.write(('original failure ' * 512 + '原反例').encode())
        output.write(b'"],"cost":17}')
    tracemalloc.start()
    try:
        with ExitStack() as opened:
            saved = storage._JsonSource(source, opened)
            fields = saved.members(keys={'cost'})
            assert fields['cost'][0] == 17
            _, peak = tracemalloc.get_traced_memory()
            assert peak < 1024 * 1024
    finally:
        tracemalloc.stop()


def test_complete_restore_bounds_temporary_JSON_memory_and_keeps_all_values(tmp_path):
    source = tmp_path / 'research_state.json'
    with source.open('w') as output:
        output.write('{"operator_results":[')
        for index in range(128):
            if index:
                output.write(',')
            json.dump({'index': index, 'failure': 'retained original failure ' * 2048,
                       'costs': {'compute': index}, 'note': '原反例'}, output,
                      ensure_ascii=False)
        output.write('],"paid":97,"cursor":{"layer":1,"offset":1}}')
    tracemalloc.start()
    try:
        restored = storage.read_json_checkpoint(source)
        current, peak = tracemalloc.get_traced_memory()
        # C decoding owns at most one bounded byte slice and its Unicode
        # transport, including four-byte Unicode for actual source strings.
        assert peak - current < 5 * storage._JSON_DECODE_CHUNK
    finally:
        tracemalloc.stop()
    assert restored['paid'] == 97
    assert restored['cursor'] == {'layer': 1, 'offset': 1}
    assert len(restored['operator_results']) == 128
    assert restored == json.loads(source.read_text())


def test_selected_leading_questions_stop_before_later_history(tmp_path, monkeypatch):
    path = tmp_path / 'research_state.json'
    questions = [{'id': 'actual parent revision', 'cursor': [1, 3]}]
    prefix = json.dumps({'questions': questions})[:-1]
    path.write_text(prefix + ', "operator_results":[' + '0,' * 100000 + '0]}')
    locate = storage._json_value_end

    def leading_fields_only(text, start):
        assert start < len(prefix), 'Do not scan later history after all requested fields were found'
        return locate(text, start)

    monkeypatch.setattr(storage, '_json_value_end', leading_fields_only)
    assert storage.read_json_checkpoint(path, keys={'questions'}) == {'questions': questions}


def test_filtered_reads_keep_full_and_later_member_lookups_correct(tmp_path):
    path = tmp_path / 'research_state.json'
    value = {'questions': [{'id': 'parent'}], 'later': [1, 2, 3], 'status': 'interrupted'}
    path.write_text(json.dumps(value))
    with ExitStack() as opened:
        source = storage._JsonSource(path, opened)
        assert set(source.members(keys={'questions'})) == {'questions'}
        assert source.members(keys={'status'})['status'][0] == 'interrupted'
        assert list(source.members()) == list(value)
        assert set(source.members(keys={'questions'})) == {'questions'}
        assert set(source.members(keys={'questions', 'missing'})) == {'questions'}
        assert source.members(keys=set()) == {}


def test_complete_restore_uses_file_end_without_first_scanning_the_root(tmp_path, monkeypatch):
    path = tmp_path / 'research_state.json'
    value = {'questions': [{'id': 'parent'}], 'paid': 97, 'failure': '原反例🧪'}
    path.write_text(' \n' + json.dumps(value, ensure_ascii=False) + '\n \t')

    def no_boundary_scan(*args):
        pytest.fail('A bounded complete root decodes directly using the file end')

    monkeypatch.setattr(storage, '_json_value_end', no_boundary_scan)
    assert storage.read_json_checkpoint(path) == value


def shared_filesystem(monkeypatch):
    copied = []

    def clone(source, destination):
        shutil.copyfile(source, destination)
        return True

    def clone_range(destination, operation, arguments):
        assert operation == 0x4020940D
        descriptor, source, length, target = struct.unpack('qQQQ', arguments)
        assert source % 4096 == target % 4096 == length % 4096 == 0
        copied.append((source, length, target))
        os.pwrite(destination, os.pread(descriptor, length, source), target)

    monkeypatch.setattr(storage, '_clone_file', clone)
    monkeypatch.setattr(storage.fcntl, 'ioctl', clone_range)
    return copied


def test_no_reflink_copies_unchanged_aligned_fields_after_changed_prefix(tmp_path, monkeypatch):
    path = tmp_path / 'metadata.json'
    previous = {'status': 'old', 'retained': {'second': list(range(20000)),
        'first': '原始反例'}, 'tail': 'preserved'}
    path.write_text(json.dumps(previous, ensure_ascii=False, indent=2))
    current = {**previous, 'status': 'new'}
    copied_ranges = []

    def no_range_clone(descriptor, operation, arguments):
        assert operation == 0x4020940D
        copied_ranges.append(arguments)
        raise OSError(errno.EOPNOTSUPP, 'filesystem has no reflinks')

    monkeypatch.setattr(storage, '_clone_file', lambda *args: False)
    monkeypatch.setattr(storage.fcntl, 'ioctl', no_range_clone)
    storage.write_json_checkpoint(path, current, ensure_ascii=False)
    assert copied_ranges
    assert b'\x00' not in path.read_bytes()
    restored = json.loads(path.read_text())
    assert storage._json_values_equal(restored, current)
    assert list(restored['retained']) == ['second', 'first']
    before = path.stat()
    storage.write_json_checkpoint(path, current, ensure_ascii=False)
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_nested_source_positions_are_lazy_and_preserve_order(tmp_path):
    source = tmp_path / 'research_state.json'
    original = {'operator_results': [{'feedback': {
        'roles': {'second': 2, 'first': 1},
        'evaluations': [{'path': 'model.py', 'residuals': list(range(20000)),
                         'error': '保留原始失败'}]}}]}
    source.write_text(json.dumps(original, ensure_ascii=False, indent=2))
    with ExitStack() as opened:
        saved = storage._JsonSource(source, opened)
        operators = saved.members()['operator_results']
        execution = saved.members(operators)[0]
        feedback = saved.members(execution)['feedback']
        evaluations = saved.members(feedback)['evaluations']
        evaluation = saved.members(evaluations)[0]
        assert all(node[0] is None for node in (operators, execution, feedback,
                                               evaluations, evaluation))
        roles = saved.members(feedback)['roles']
        assert list(saved.members(roles)) == ['second', 'first']
        assert json.loads(saved.text[evaluation[1]:evaluation[2]]) == (
            original['operator_results'][0]['feedback']['evaluations'][0])
        assert source.read_bytes()[saved.byte_position(evaluation[1]):
                                   saved.byte_position(evaluation[2])] == (
                                       saved.text[evaluation[1]:evaluation[2]])


def test_unrequested_large_source_field_is_located_without_decoding_it(tmp_path, monkeypatch):
    source = tmp_path / 'training.json'
    value = {'unrequested': [{'text': '[]{} \\" 保留', 'rows': list(range(20000))}],
             'wanted': {'rows': [1, 2], 'text': 'escaped quote: " and slash: \\'}}
    source.write_text(json.dumps(value, ensure_ascii=False))
    decode = json.JSONDecoder.raw_decode

    def no_unrequested_tree(decoder, text, idx=0):
        if text[idx] == '[':
            pytest.fail('The unrequested source list must not be decoded.')
        return decode(decoder, text, idx)

    monkeypatch.setattr(json.JSONDecoder, 'raw_decode', no_unrequested_tree)
    with ExitStack() as opened:
        saved = storage._JsonSource(source, opened)
        fields = saved.members(keys={'wanted'})
        assert set(fields) == {'wanted'}
        node = fields['wanted']
        assert node[0] is None
        assert json.loads(saved.text[node[1]:node[2]]) == value['wanted']


def test_container_boundaries_do_not_decode_duplicate_subtrees(tmp_path, monkeypatch):
    source = tmp_path / 'research_state.json'
    value = {'kept': {'list': [{'quoted': '[]{} " \\ 中文', 'rows': list(range(20000))}],
                      'flag': True, 'loss': float('nan')}}
    source.write_text(json.dumps(value, ensure_ascii=False))
    with ExitStack() as opened:
        saved = storage._JsonSource(source, opened)
        parent = saved.members()['kept']
        decode = json.JSONDecoder.raw_decode

        def no_duplicate_tree(decoder, text, idx=0):
            if text[idx] in '[{':
                pytest.fail('A parsed parent already owns this child tree.')
            return decode(decoder, text, idx)

        monkeypatch.setattr(json.JSONDecoder, 'raw_decode', no_duplicate_tree)
        children = saved.members(parent)
        first = saved.members(children['list'])[0]
        assert parent[0] is children['list'][0] is first[0] is None
        assert saved.text[first[1]:first[2]].startswith(b'{')
        assert children['flag'][0] is True
        assert math.isnan(children['loss'][0])


@pytest.mark.parametrize('separators', [(',', ':'), (', ', ': ')])
def test_bounded_saved_record_exact_encoding_skips_recursive_comparison(
        tmp_path, monkeypatch, separators):
    copied = shared_filesystem(monkeypatch)
    source, target = tmp_path / 'old.json', tmp_path / 'research_state.json'
    record = {'program': [{'op': 'add', 'args': [index, index + 1]}
                          for index in range(3000)], 'failure': 'original counterexample'}
    source.write_text(json.dumps({'header': 'old', 'record': record}, separators=separators))
    members = storage._JsonSource.members

    def no_record_walk(instance, node=None, **kwargs):
        if node is not None:
            pytest.fail('An exactly encoded bounded record needs no recursive node walk.')
        return members(instance, node, **kwargs)

    monkeypatch.setattr(storage._JsonSource, 'members', no_record_walk)
    current = {'header': 'a changed header', 'record': record}
    storage.write_json_checkpoint(target, current, base=source, indent=None)
    assert json.loads(target.read_text()) == current
    assert copied


def test_source_json_trees_and_read_files_are_released_before_fsync(tmp_path, monkeypatch):
    shared_filesystem(monkeypatch)
    source, target = tmp_path / 'old.json', tmp_path / 'research_state.json'
    previous = {'stage': 'before', 'evaluations': list(range(20000)), 'cost': 17}
    target.write_text(json.dumps(previous))
    source.write_bytes(target.read_bytes())  # Also exercise a filtered identical source.
    changed = {**previous, 'stage': 'after the complete retained comparison', 'cost': 18}
    retained, commits = [], []
    initialize, synchronize = storage._JsonSource.__init__, os.fsync

    def remember(instance, path, opened):
        initialize(instance, path, opened)
        retained.append(instance)

    def commit(descriptor):
        assert len(retained) == 2
        assert all(item.text is None and item.fields is None and item.file.closed
                   for item in retained)
        # The replacement is still uncommitted; closing read handles must not
        # change the caller's input or move the atomic replacement boundary.
        assert json.loads(target.read_text()) == previous
        assert changed['evaluations'] == previous['evaluations']
        commits.append(descriptor)
        synchronize(descriptor)

    monkeypatch.setattr(storage._JsonSource, '__init__', remember)
    monkeypatch.setattr(storage.os, 'fsync', commit)
    storage.write_json_checkpoint(target, changed, base=source, indent=None)
    assert len(commits) == 1
    assert json.loads(target.read_text()) == changed
    assert json.loads(source.read_text()) == previous


def test_fsync_enomem_propagates_and_preserves_previous_checkpoint(tmp_path, monkeypatch):
    shared_filesystem(monkeypatch)
    target = tmp_path / 'research_state.json'
    previous = {'stage': 'before', 'evaluations': list(range(20000)), 'cost': 17}
    target.write_text(json.dumps(previous))
    before = target.read_bytes()
    changed = {**previous, 'stage': 'after the complete retained comparison', 'cost': 18}
    commits = []

    def commit(descriptor):
        commits.append(descriptor)
        raise OSError(errno.ENOMEM, 'Cannot allocate memory')

    monkeypatch.setattr(storage.os, 'fsync', commit)
    with pytest.raises(OSError) as failure:
        storage.write_json_checkpoint(target, changed, indent=None)
    assert failure.value.errno == errno.ENOMEM
    assert len(commits) == 1
    assert target.read_bytes() == before
    assert not target.with_name(target.name + '.writing').exists()
    assert changed['evaluations'] == previous['evaluations']


def test_moved_large_values_skip_encoding_and_preserve_complete_changed_state(tmp_path, monkeypatch):
    copied = shared_filesystem(monkeypatch)
    source, target = tmp_path / 'old.json', tmp_path / 'research_state.json'
    old = {'questions': [{'parent': 'retained', 'rows': list(range(20000))}],
        'operator_results': [{'failure': 'counterexample preserved', 'evidence': list(range(30000))}],
        'costs': {'compute': 17}}
    source.write_text(json.dumps(old))
    previous = source.read_bytes()
    changed = deepcopy(old)
    changed['questions'][0]['parent'] = 'a much longer future library path'
    changed['costs']['compute'] += 1
    encode = json.JSONEncoder.encode

    def no_reencoding(encoder, value):
        if isinstance(value, list) and len(value) > 10000:
            pytest.fail('the equal retained value must be cloned without encoding again')
        return encode(encoder, value)

    monkeypatch.setattr(json.JSONEncoder, 'encode', no_reencoding)
    storage.write_json_checkpoint(target, changed, base=source, indent=None)
    assert json.loads(target.read_text()) == changed
    assert source.read_bytes() == previous
    assert sum(length for _, length, _ in copied) > 200000
    assert any(first != destination for first, _, destination in copied)
    before = target.stat()
    storage.write_json_checkpoint(target, changed, indent=None)
    assert (target.stat().st_ino, target.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_final_record_reuses_actual_training_and_future_final_reuses_local_active(tmp_path, monkeypatch):
    shared_filesystem(monkeypatch)
    source, training = tmp_path / 'source_state.json', tmp_path / 'source_training.json'
    record = {'failure': 'retained', 'evidence': list(range(20000))}
    source.write_text(json.dumps({'method_results': [record]}))
    completed = {'method_results': [record, {**record, 'failure': 'distinct last failure'}]}
    training.write_text(json.dumps(completed, separators=(',', ':')))
    active, final = tmp_path / 'active.json', tmp_path / 'training.json'
    storage.retain_checkpoint_sources(active, source, training)
    storage.write_json_checkpoint(active, completed)
    assert json.loads(active.read_text()) == completed
    new = {**completed, 'counterexamples': [{'unique': True}]}
    storage.write_json_checkpoint(active, new)
    storage.write_json_checkpoint(final, {**new, 'status': 'failed'})
    assert json.loads(final.read_text()) == {**new, 'status': 'failed'}
    assert json.loads(active.read_text()) == new
    # Original failure, last failure and the newly added one all remain distinct.
    assert [item['failure'] for item in json.loads(final.read_text())['method_results']] == [
        'retained', 'distinct last failure']


def test_appended_comparison_shares_paid_score_prefix_and_retains_new_failure(tmp_path, monkeypatch):
    copied = shared_filesystem(monkeypatch)
    source, target = tmp_path / 'source.json', tmp_path / 'research_state.json'
    scores = [{'path': f'model_{index}.py', 'mse': index / 100,
        'paid_residuals': {'examples': [{'target': 1., 'prediction': 2.,
            'history': [0.1] * 400}]},
        'paid_score_reuse': {'assessment_ref': {'file': f'assessment_{index}.json',
            'json_path': ['result']}}} for index in range(120)]
    previous = {'operator_results': [{'index': 0, 'evaluations': scores,
        'costs': {'compute': 97}, 'counterexample': 'original failure'}]}
    source.write_text(json.dumps(previous))
    current = deepcopy(previous)
    appended = {'index': 1, 'evaluations': [*deepcopy(scores),
        {'path': 'new_failed.py', 'returncode': 1, 'stderr': 'actual failure'}],
        'costs': {'compute': 194}, 'counterexample': 'new failure'}
    current['operator_results'].append(appended)
    encode = json.JSONEncoder.encode

    def no_reencoding(encoder, value):
        if value is appended:
            pytest.fail('an appended comparison must share its retained score prefix')
        return encode(encoder, value)

    monkeypatch.setattr(json.JSONEncoder, 'encode', no_reencoding)
    storage.write_json_checkpoint(target, current, base=source, indent=None)
    restored = json.loads(target.read_text())
    assert restored == current
    assert restored['operator_results'][1]['evaluations'][-1]['stderr'] == 'actual failure'
    assert restored['operator_results'][1]['costs']['compute'] == 194
    assert json.loads(source.read_text()) == previous
    # Both the old execution and the new execution's score prefix share data.
    assert sum(length for _, length, _ in copied) > 2 * 200000


def test_equal_retained_history_shares_one_complete_range(tmp_path, monkeypatch):
    copied = shared_filesystem(monkeypatch)
    source, target = tmp_path / 'source.json', tmp_path / 'research_state.json'
    history = [{'operator': 'research_choice', 'evaluations': [
        {'model': f'model_{index}.py', 'residuals': list(range(1000)),
         'retention_decision': {'status': 'incumbent_retained'}}
        for index in range(30)], 'compute_cost': turn}
        for turn in range(4)]
    original = {'operator_results': history, 'compute_cost': 97}
    source.write_text(json.dumps(original))
    current = {**original, 'compute_cost': 98}
    storage.write_json_checkpoint(target, current, base=source, indent=None)
    assert json.loads(target.read_text()) == current
    # Equal old owners need no score-by-score rewrite to obtain sharing.
    # Changed/appended owners still use the existing finer-grained path.
    assert max(length for _, length, _ in copied) > len(json.dumps(history)) * .9


def test_changed_score_between_equal_runs_does_not_reuse_an_old_decision(tmp_path, monkeypatch):
    copied = shared_filesystem(monkeypatch)
    source, target = tmp_path / 'source.json', tmp_path / 'research_state.json'
    scores = [{'path': f'model_{index}.py', 'residuals': [index] * 700,
        'retention_decision': {'status': 'incumbent_retained'}} for index in range(120)]
    source.write_text(json.dumps({'evaluations': scores}, separators=(',', ':')))
    current = deepcopy(scores)
    current[60]['retention_decision'] = {'status': 'candidate_retained', 'cost': 17}
    storage.write_json_checkpoint(target, {'evaluations': current}, base=source, indent=None)
    restored = json.loads(target.read_text())
    assert restored['evaluations'] == current
    assert restored['evaluations'][60]['retention_decision']['status'] == 'candidate_retained'
    assert sum(length for _, length, _ in copied) > 200000


def test_inserted_failure_and_removed_score_keep_the_actual_array_order(tmp_path, monkeypatch):
    shared_filesystem(monkeypatch)
    source, target = tmp_path / 'source.json', tmp_path / 'research_state.json'
    scores = [{'path': f'model_{index}.py', 'residuals': [index] * 700}
        for index in range(120)]
    source.write_text(json.dumps({'evaluations': scores}))
    changed = [*deepcopy(scores[:40]),
        {'path': 'failed.py', 'error': 'unique execution failure', 'cost': 97},
        *deepcopy(scores[41:])]
    storage.write_json_checkpoint(target, {'evaluations': changed}, base=source, indent=None)
    assert json.loads(target.read_text())['evaluations'] == changed
    assert json.loads(source.read_text())['evaluations'] == scores


def test_changed_large_execution_and_short_common_tail_use_linear_comparisons(tmp_path, monkeypatch):
    shared_filesystem(monkeypatch)
    source, target = tmp_path / 'source.json', tmp_path / 'research_state.json'
    tail = [{'path': f'failed_{index}.py', 'returncode': 1,
        'error': 'retained execution failure', 'compute_cost': index}
        for index in range(200)]
    previous = {'operator_results': [{'index': 0, 'diagnosis': 'old',
        'evaluations': list(range(20000))}, *tail]}
    source.write_text(json.dumps(previous))
    current = deepcopy(previous)
    current['operator_results'][0]['diagnosis'] = 'new'
    equal = storage._json_values_equal
    comparisons = 0

    def counted(saved, value, allow_nan=True):
        nonlocal comparisons
        if isinstance(value, dict) and value.get('path', '').startswith('failed_'):
            comparisons += 1
        return equal(saved, value, allow_nan)

    monkeypatch.setattr(storage, '_json_values_equal', counted)
    storage.write_json_checkpoint(target, current, base=source, indent=None)
    assert json.loads(target.read_text()) == current
    assert json.loads(source.read_text()) == previous
    assert comparisons <= 2 * len(tail), comparisons


def test_json_transport_tuple_and_existing_nan_failure_are_preserved(tmp_path, monkeypatch):
    copied = shared_filesystem(monkeypatch)
    source, target = tmp_path / 'source.json', tmp_path / 'research_state.json'
    rows = [(index, index + 1) for index in range(20000)]
    source.write_text(json.dumps({'rows': rows, 'failed_loss': float('nan')}))
    storage.write_json_checkpoint(target, {'rows': rows, 'failed_loss': float('nan')}, base=source, indent=None)
    restored = json.loads(target.read_text())
    assert restored['rows'] == [list(row) for row in rows]
    assert math.isnan(restored['failed_loss']) and copied
    with pytest.raises(ValueError):
        storage.write_json_checkpoint(tmp_path / 'training.json', {'failed_loss': float('nan')})


def test_failed_range_replacement_keeps_original_and_no_partial_checkpoint(tmp_path, monkeypatch):
    shared_filesystem(monkeypatch)
    target = tmp_path / 'research_state.json'
    target.write_text(json.dumps({'header': 'old', 'rows': list(range(20000))}))
    original = target.read_bytes()

    def failed(_descriptor):
        raise OSError('storage full')

    monkeypatch.setattr(storage.os, 'fsync', failed)
    with pytest.raises(OSError, match='storage full'):
        storage.write_json_checkpoint(target, {'header': 'new changed header', 'rows': list(range(20000))}, indent=None)
    assert target.read_bytes() == original
    assert not target.with_name(target.name + '.writing').exists()


@pytest.mark.parametrize('storage_errno', [errno.EIO, errno.ENOSPC])
def test_range_storage_error_retains_operation_details_without_copy_retry(
        tmp_path, monkeypatch, storage_errno):
    shared_filesystem(monkeypatch)
    target = tmp_path / 'research_state.json'
    target.write_text(json.dumps({'header': 'old', 'rows': list(range(20000))}))
    original = target.read_bytes()
    attempted = []

    def failed(descriptor, operation, arguments):
        attempted.append(struct.unpack('qQQQ', arguments))
        raise OSError(storage_errno, 'actual storage failure')

    def forbidden_copy(*args, **kwargs):
        pytest.fail('An I/O or capacity failure must not enter the copy fallback.')

    monkeypatch.setattr(storage.fcntl, 'ioctl', failed)
    monkeypatch.setattr(storage.os, 'pwrite', forbidden_copy)
    with pytest.raises(OSError) as caught:
        storage.write_json_checkpoint(target,
            {'header': 'a changed header', 'rows': list(range(20000))}, indent=None)
    assert caught.value.errno == storage_errno
    assert len(attempted) == 1
    _, first, length, destination = attempted[0]
    assert first % 4096 == length % 4096 == destination % 4096 == 0
    message = str(caught.value)
    assert str(target) in message and str(target) + '.writing' in message
    assert f'offset={first} length={length}' in message
    assert f'offset={destination}' in message and 'size=' in message
    assert target.read_bytes() == original
    assert not target.with_name(target.name + '.writing').exists()


def test_nested_component_dictionary_order_is_preserved_inside_a_large_list(tmp_path, monkeypatch):
    shared_filesystem(monkeypatch)
    source, target = tmp_path / 'source.json', tmp_path / 'research_state.json'
    old = {'bank': [{'components': {'later': 2, 'first': 1}, 'paid_rows': list(range(20000))}]}
    source.write_text(json.dumps(old))
    current = deepcopy(old)
    current['bank'][0]['components'] = {'first': 1, 'later': 2}
    storage.write_json_checkpoint(target, current, base=source, indent=None)
    assert list(json.loads(target.read_text())['bank'][0]['components']) == ['first', 'later']
    assert list(json.loads(source.read_text())['bank'][0]['components']) == ['later', 'first']


def test_strict_old_writer_nan_contract_also_applies_to_shared_large_values(tmp_path, monkeypatch):
    shared_filesystem(monkeypatch)
    source, target = tmp_path / 'old.json', tmp_path / 'new.json'
    value = {'bank': [{'loss': float('nan'), 'paid_rows': list(range(20000))}]}
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Out of range'):
        storage.write_json_checkpoint(target, value, base=source, allow_nan=False)
    assert not target.exists() and not target.with_name(target.name + '.writing').exists()


def test_question_read_does_not_decode_unrequested_large_research_bank(tmp_path, monkeypatch):
    path = tmp_path / 'research_state.json'
    questions = [{'id': '父修订', 'native_condition_research': [{'cursor': [1, 1]}]}]
    path.write_text(json.dumps({'theory_index': {'retained': 'x' * (8 * 1024 * 1024)},
                               'questions': questions, 'costs': {'compute_cost': 123}}))
    loads = storage.json.loads

    def small_fields_only(value, *args, **kwargs):
        assert len(value) < 1024, 'Reading questions must not decode the unrelated bank'
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(storage.json, 'loads', small_fields_only)
    assert storage.read_json_checkpoint(path, keys={'questions'}) == {'questions': questions}
