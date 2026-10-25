"""The complete comparison is learned; intermediate decisions remain evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace

from hegel_machine.cognition.theory_consolidation import _revision_entries
from test_revision_relations import binary, constant, feature


def test_complete_step_replaces_suffix_for_learning_without_losing_ledger(tmp_path):
    before, middle = feature(), binary('add', feature(), constant(1))
    after = binary('mul', middle, constant(2))
    path = tmp_path / 'source.json'
    path.write_text(json.dumps({'before': before.to_dict(), 'middle': middle.to_dict(),
                                'after': after.to_dict()}))
    ref = lambda key: {'file': str(path), 'json_path': [key]}
    member_refs = [ref('retention_0'), ref('retention_1')]
    step = {'kind': 'completed_research_comparison', 'status': 'completed',
        'retained_outcome': 'parent_changed', 'before_ref': ref('before'),
        'source_ref': ref('after'), 'action_ref': ref('comparison'),
        'comparison_pool_ref': ref('evaluations'), 'member_retention_refs': member_refs,
        'scope': {'same_paid_rows': True, 'evaluated_rows': 97},
        'source_comparison': {'status': 'UNKNOWN', 'reason': 'no_saved_scores_in_fixture'},
        'cost_refs': [ref('compute_cost')],
        'cost_scope': 'whole_research_choice_not_additive_per_revision',
        'producer_link': {'status': 'UNKNOWN', 'reason': 'no_saved_producer_in_fixture'}}
    unchanged = {**deepcopy(step), 'retained_outcome': 'parent_unchanged',
                 'before_ref': ref('after'), 'action_ref': ref('comparison_without_adoption')}
    unresolved = {'status': 'UNKNOWN', 'reason': 'incomplete_saved_comparison',
                  'action_ref': ref('incomplete_comparison')}
    episode = {'episode_id': 'actual_source', 'source_id': 'physical_source', 'theory_id': 'winner',
        'cost_refs': [ref('source_costs')], 'cost_scope': 'whole_Common_source_account_not_additive_per_question',
        'actual_changes': [{'before_ref': ref('middle'), 'source_ref': ref('after')}],
        'source_revision_steps': [step, unchanged, unresolved]}
    original = deepcopy(episode)
    agent = SimpleNamespace(workspace=tmp_path, questions=[], revisions=[], observations=[], theory_index={})
    entry, = _revision_entries(agent, [episode], {'winner': {'program_source_ref': ref('after')}}, [])
    assert entry['before_program'] == before.to_dict()
    assert entry['after_program'] == after.to_dict()
    assert entry['source_id'] == 'physical_source'
    assert entry['source_comparison']['research_step']['member_retention_refs'] == member_refs
    assert entry['source_comparison']['research_step']['cost_refs'] == [ref('compute_cost')]
    assert entry['cost_refs'] == [ref('source_costs')]
    assert entry['source_outcomes']['source_revision_steps'] == [step, unchanged, unresolved]
    assert episode == original


def test_explicit_incomplete_step_does_not_fall_back_to_partial_suffix(tmp_path):
    episode = {'episode_id': 'source', 'source_id': 'one_source', 'theory_id': 'winner',
        'actual_changes': [{'before_ref': {'file': 'missing_partial.json'},
                            'source_ref': {'file': 'missing_winner.json'}}],
        'source_revision_steps': [{'status': 'UNKNOWN', 'reason': 'truncated_roster'}]}
    agent = SimpleNamespace(workspace=tmp_path, questions=[], revisions=[], observations=[], theory_index={})
    assert _revision_entries(agent, [episode], {'winner': {}}, []) == []


def test_old_cached_consolidation_without_step_inputs_recomputes_once_and_keeps_costs(tmp_path):
    from test_retained_revision_triggers import retained_source, build
    from test_theory_consolidation import agent_scene

    sources = [retained_source(tmp_path / f'source_{i}', i) for i in range(2)]
    references = [source[3] for source in sources]
    agent, lab = agent_scene(tmp_path / 'target')
    first = build(agent, lab, references)
    theme = 'actual_retained_edits'
    prior = agent.theory_consolidation['executions'][theme]
    assert prior['request_basis']['source_revision_steps']
    # A saved result from before the complete-step intake had no such inputs.
    # Its source files, budget and requested theory IDs can all be unchanged.
    del prior['request_basis']['source_revision_steps']
    paid = deepcopy(agent.costs)
    second = build(agent, lab, references)
    assert not second['cache_hit'] and second['new_compute_cost'] > 0
    assert agent.costs['compute_cost'] == paid['compute_cost'] + second['new_compute_cost']
    assert second['frozen_candidates'] == first['frozen_candidates']
    spent = deepcopy(agent.costs)
    third = build(agent, lab, references)
    assert third['cache_hit'] and third['new_compute_cost'] == 0 and agent.costs == spent
    # A local comparison is another source action even when it has no new
    # observations, Program version or native author round. Its incomplete
    # result remains unresolved, but it cannot reuse the prior action intake.
    agent.events.append({'action': 'run', 'arguments': {'operator': 'research_choice', 'models': []},
        'feedback': {'operator': 'research_choice', 'research_question_id': 'local_unfinished',
                     'question_meaning_version': 1, 'paid_rows': 0, 'evaluations': [],
                     'error': 'comparison interrupted'}})
    fourth = build(agent, lab, references)
    assert not fourth['cache_hit']
    assert agent.theory_consolidation['executions'][theme]['request_basis']['local_comparison_refs']
    local = next(episode for episode in agent.theory_consolidation['episodes']
                 if episode['episode_id'] == 'local_unfinished@meaning1')
    assert local['source_revision_steps'][0]['status'] == 'UNKNOWN'
    assert not fourth['author_work']  # unrelated old gaps do not block valid selected sources
    assert lab.paid == agent.costs['model_calls'] == 0
