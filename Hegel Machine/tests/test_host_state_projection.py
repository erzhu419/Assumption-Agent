"""Inline metadata cutoffs must not encode the complete retained score bank."""

import json

import pytest

from hegel_machine.cognition import code_provider_context as context


@pytest.mark.parametrize('value', [None, False, 0, -1.5, float('inf'), '', 'a' * 11999,
    '\u4e2d' * 2100, '\U0001f642' * 1100, '\\"\n' * 2500,
    [1, 2, {'scope': ['a', None, False]}], {'nested': ['x' * 6000, 'y' * 6000]},
    {0: False, None: [True], 2.5: 'measured'}, ({'cost': 1}, 'kept')])
def test_bounded_inline_check_keeps_the_exact_existing_json_cutoff(value):
    length = len(json.dumps(value))
    for limit in (0, 2, 5, 11999, 12000, length - 1, length, length + 1):
        if limit >= 0:
            assert context._fits_inline_json(value, limit) == (length <= limit)


def test_large_ended_feedback_is_referenced_without_serializing_its_payload(monkeypatch):
    payload = {'observed': ['x' * 100000], 'status': 'complete',
        'costs': {'source': 'y' * 100000, 'compute_cost': 7}}
    encode = json.dumps
    calls = []
    def bounded(value, *args, **kwargs):
        assert value is not payload
        assert not isinstance(value, str) or len(value) <= 12000
        calls.append(type(value).__name__)
        return encode(value, *args, **kwargs)
    monkeypatch.setattr(context.json, 'dumps', bounded)
    assert not context._fits_inline_json(payload)
    reference = context._saved_context_reference(payload, ['method_results', 33, 'feedback'],
        'research_state.json')
    assert reference['status'] == 'complete'
    assert reference['costs']['json_path'] == ['method_results', 33, 'feedback', 'costs']
    assert payload['costs']['compute_cost'] == 7
