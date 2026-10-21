"""Unfinished typed families are budget unknowns, not absent peers."""

from copy import deepcopy

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.theory_induction import consolidate_theories
from test_theory_induction import episode


def test_known_peer_budget_unknown_keeps_completed_family_and_its_counterexample():
    left, right = episode(), episode('b', 'u')
    left['counterexamples'] = [{'source_ref': {'file': 'retained.json', 'json_path': ['counterexample']},
        'observation': {'features': {'x': 1.}, 'target': 99., 'history': []}}]
    later_left, later_right = episode('c'), episode('d', 'u')
    later_left['program'] = Program('lag', name='x', lag=1).to_dict()
    later_right['program'] = Program('lag', name='u', lag=1).to_dict()
    singleton = episode('e')
    singleton['program'] = Program('const', value=1.).to_dict()
    sources = [left, right, later_left, later_right, singleton]
    original = deepcopy(sources)
    # 18 source visits + 10 reconstruction nodes + 6 retained-check steps:
    # all groups are known, and only the first family has completed.
    partial = consolidate_theories(sources, allocation={'compute': 34})
    complete = consolidate_theories(sources, allocation={'compute': 36})
    assert partial['status'] == 'UNKNOWN' and partial['budget_exhausted']
    assert partial['actual_compute_cost'] == 34
    retained, = partial['candidates']
    assert retained == complete['candidates'][0]
    assert retained['counterexample_checks'][0]['status'] == 'failed'
    assert not retained['active_library']
    obligations = {request['episode_id']: request for request in partial['author_work']}
    for identifier in ('c', 'd'):
        assert obligations[identifier]['status'] == 'UNKNOWN'
        assert obligations[identifier]['reason'] == 'induction_budget_exhausted'
    assert obligations['e']['reason'] == 'no_compatible_ordered_typed_peer'
    assert complete['status'] == 'completed' and not complete['budget_exhausted']
    assert [request['episode_id'] for request in complete['author_work']] == ['e']
    assert sources == original
    assert partial['new_observations'] == partial['llm_calls'] == 0


def test_incomplete_source_scan_cannot_declare_that_an_unseen_peer_is_absent():
    sources = [episode(), episode('b', 'u')]
    # The first affine source consumes one source visit and five nodes.
    # Its second, compatible source has not yet been scanned.
    result = consolidate_theories(sources, allocation={'compute': 6})
    assert result['status'] == 'UNKNOWN' and result['budget_exhausted']
    assert result['actual_compute_cost'] == 6 and not result['candidates']
    request, = result['author_work']
    assert request['episode_id'] == 'a'
    assert request['status'] == 'UNKNOWN'
    assert request['reason'] == 'induction_budget_exhausted'
