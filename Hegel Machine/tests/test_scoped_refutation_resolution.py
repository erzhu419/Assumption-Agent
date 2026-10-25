"""Exact failure indexing preserves all source versions and reference order."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import question_routing as routing
from hegel_machine.cognition.language import Program


CONTEXT = {'scene': 'old_paid_scope'}
ROLES = {'u': {'role': 'input', 'unit': 'V', 'time': 'current'}}


def theory(identifier, value=1., **extra):
    return {'id': identifier, 'version': 1, 'program': Program('const', value=value).to_dict(),
        'role_signature': deepcopy(ROLES), 'output_unit': 'V', 'context_ref': deepcopy(CONTEXT),
        'support': {'status': 'candidate'}, **extra}


def agent(tmp_path, theories, transfers=(), questions=()):
    return SimpleNamespace(workspace=tmp_path, questions=list(questions), events=[], relation_research={},
        theory_index={item['id']: item for item in theories}, theory_consolidation={'transfers': list(transfers)})


def failed_transfer(identifier='failed', *, version=1, binding=None, source='paid'):
    return {'theory_id': identifier, 'theory_ref': {'id': identifier, 'version': version},
        'binding': binding or {'context_ref': deepcopy(CONTEXT)},
        'status': 'refuted', 'checks': [{'status': 'failed', 'context_ref': deepcopy(CONTEXT),
            'source_ref': {'file': source + '.json', 'json_path': [0]}}]}


def check_ref(index):
    return {'file': 'shared/theory_library.json', 'json_path': ['transfers', index, 'checks', 0]}


def test_history_and_modern_legacy_transfers_keep_complete_original_reference_order(tmp_path):
    binding = {'context_ref': deepcopy(CONTEXT), 'legacy': 'retained'}
    previous = theory('failed', compatible=False, consolidation_ref=deepcopy(binding))
    current = theory('failed', version=2, compatible=False, history=[previous])
    legacy = failed_transfer(binding=binding, source='legacy')
    legacy.pop('theory_ref')
    newest = failed_transfer(version=2, source='newest')
    oldest = failed_transfer(version=1, source='oldest')
    proxy = agent(tmp_path, [current, theory('alias')], [legacy, newest, oldest])
    untouched = deepcopy((proxy.theory_index, proxy.theory_consolidation))
    result = routing.resolved_theories(proxy)
    scope = result[1]['scope_refutation']
    assert scope['theory_refs'] == [
        {'file': 'research_state.json', 'json_path': ['theory_index', 'failed', 'history', 0], 'version': 1},
        {'file': 'research_state.json', 'json_path': ['theory_index', 'failed'], 'version': 2}]
    assert scope['counterexample_refs'] == [check_ref(0), {'file': 'legacy.json', 'json_path': [0]},
        check_ref(2), {'file': 'oldest.json', 'json_path': [0]},
        check_ref(1), {'file': 'newest.json', 'json_path': [0]}]
    assert routing.scoped_numeric_refutation(proxy, theory('incoming')) == scope
    assert (proxy.theory_index, proxy.theory_consolidation) == untouched
    scope['counterexample_refs'][0]['json_path'].append('edited_return')
    again = routing.scoped_numeric_refutation(proxy, theory('incoming'))
    assert again['counterexample_refs'][0] == check_ref(0)


def test_hash_collisions_do_not_merge_programs_or_drop_scope_contracts(tmp_path, monkeypatch):
    monkeypatch.setattr(Program, '__hash__', lambda self: 1)
    proxy = agent(tmp_path, [theory('failed', compatible=False), theory('same'), theory('other', value=2.),
        theory('other_scope', context_ref={'scene': 'elsewhere'}),
        theory('other_role', role_signature={'u': {'role': 'different', 'unit': 'V', 'time': 'current'}})],
        [failed_transfer()])
    result = {item['id']: item for item in routing.resolved_theories(proxy)}
    assert result['same']['compatible'] is False
    assert all('scope_refutation' not in result[name] for name in ('other', 'other_scope', 'other_role'))
    assert routing.scoped_numeric_refutation(proxy, theory('incoming', value=2.)) is None


def test_new_failure_version_and_source_changes_are_seen_without_global_cache(tmp_path):
    proxy = agent(tmp_path, [theory('failed', compatible=False), theory('alias')])
    assert routing.scoped_numeric_refutation(proxy, theory('incoming')) is None
    proxy.theory_consolidation['transfers'].append(failed_transfer())
    assert routing.scoped_numeric_refutation(proxy, theory('incoming'))['counterexample_refs'][0] == check_ref(0)
    proxy.theory_consolidation['transfers'][0]['theory_ref']['version'] = 2
    assert routing.scoped_numeric_refutation(proxy, theory('incoming')) is None
    proxy.theory_index['failed']['version'] = 2
    assert routing.scoped_numeric_refutation(proxy, theory('incoming'))
    proxy.theory_index['failed']['program'] = Program('const', value=3.).to_dict()
    assert routing.scoped_numeric_refutation(proxy, theory('incoming')) is None


def test_native_source_failure_is_read_from_actual_retained_bank(tmp_path):
    source = {'kind': 'native_candidate', 'question_id': 'q', 'meaning_version': 1,
              'round_index': 0, 'candidate_index': 0}
    candidate = {'status': 'candidate', 'program': Program('const', value=1.).to_dict()}
    question = {'id': 'q', 'meaning_version': 1, 'native_condition_research': [{'candidates': [candidate]}]}
    native = theory('native', source_ref=source)
    native.pop('program')
    proxy = agent(tmp_path, [native], questions=[question])
    assert routing.scoped_numeric_refutation(proxy, theory('incoming')) is None
    candidate['status'] = 'refuted'
    scope = routing.scoped_numeric_refutation(proxy, theory('incoming'))
    assert scope['counterexample_refs'] == [source]
    candidate['program'] = Program('const', value=2.).to_dict()
    assert routing.scoped_numeric_refutation(proxy, theory('incoming')) is None


def test_incoming_check_does_not_decode_unrelated_positive_programs(tmp_path, monkeypatch):
    proxy = agent(tmp_path, [theory('failed', compatible=False),
        *(theory(f'old_positive_{index}', value=float(index + 10)) for index in range(300))],
        [failed_transfer()])
    calls, decode = [], Program.from_dict

    def counted(payload):
        calls.append(payload)
        return decode(payload)

    monkeypatch.setattr(Program, 'from_dict', counted)
    assert routing.scoped_numeric_refutation(proxy, theory('incoming'))
    assert len(calls) == 2  # one actual retained failure and this incoming Program


def test_incoming_scope_does_not_decode_irrelevant_negative_programs(tmp_path, monkeypatch):
    proxy = agent(tmp_path, [theory('failed', compatible=False)], [failed_transfer()])
    incoming = theory('elsewhere', context_ref={'scene': 'independent_scope'})
    calls, decode = [], Program.from_dict

    def counted(payload):
        calls.append(payload)
        return decode(payload)

    monkeypatch.setattr(Program, 'from_dict', counted)
    assert routing.scoped_numeric_refutation(proxy, incoming) is None
    assert len(calls) == 1


def test_full_resolution_returns_independent_program_and_support_values(tmp_path):
    source = theory('retained')
    proxy = agent(tmp_path, [source])
    result = routing.resolved_theories(proxy)
    result[0]['program']['value'] = 7.
    result[0]['support']['status'] = 'edited_return'
    assert source['program']['value'] == 1.
    assert source['support']['status'] == 'candidate'
