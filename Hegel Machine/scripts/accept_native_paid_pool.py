#!/usr/bin/env python3
"""Use completed native returns to check stopping, without repeating research."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts.run_native_decision_pair import load_prepared, ref
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.object_continuation import decode_entry
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.research_graph import ResearchGraph


def reject_research(*args, **kwargs):
    raise AssertionError('A completed paid pool must not fit, predict or repeat its preparation')


def accept(root):
    comparison = load(root / 'challenger/comparison.json')
    outcome = load(root / 'challenger/outcome.json')
    if not comparison['full_comparison'] or not outcome['execution_complete']:
        raise ValueError('The stored native return has no complete comparison')
    with ExitStack() as stack:
        stack.enter_context(patch('hegel_machine.cognition.language.Program.predict', reject_research))
        stack.enter_context(patch('hegel_machine.cognition.unified_research.loss', reject_research))
        agent, frame, _ = load_prepared(root)
        unresolved_before = not agent._research_pool_explained(frame)
        stack.enter_context(patch.object(agent, '_execute_prepared_research_segment', reject_research))
        claims = [*frame['assessed'], *[decode_entry(item)
            for item in load(root / 'challenger/new_claims.json')]]
        if ([item['validation_loss'] for item in claims]
                != [item['loss'] for item in comparison['scores']]):
            raise ValueError('The retained claims do not match their paid full-pool scores')
        frame['assessed'] = claims
        work, actions, statuses = SearchBudget(0), [], {}
        for operation in ('program_replay', 'structural_research'):
            result = agent.execute_research_decision(frame, operation, work, actions)
            assert result['status'] == 'already_explained'
            assert result['claims'] == claims and not result['new_claims']
            assert not result['independent_confirmation']
            statuses[operation] = result['status']
        stack.enter_context(patch.object(agent, 'prepare_research_decision', lambda *a, **k: frame))
        ordinary = agent._research_view(frame['spec'], frame['training'], frame['validation'],
            work, actions, [], ResearchGraph(), parent=frame['parent'])
        assert unresolved_before and ordinary == claims and not actions and work.spent == 0
        selected = agent._best(claims)
    return {'source_return_ref': ref(root / 'challenger/outcome.json'),
        'source_full_pool_ref': ref(root / 'challenger/comparison.json'),
        'source_cost_ref': ref(root / 'receipt.json'),
        'unresolved_before_return': unresolved_before, 'retained_candidates': len(claims),
        'selected_training_loss': selected['training_loss'],
        'selected_working_loss': selected['validation_loss'],
        'explicit_entry_statuses': statuses, 'ordinary_entry_retained_complete_pool': True,
        'new_compute': 0, 'new_fits': 0, 'new_predictions': 0, 'new_observations': 0,
        'independent_confirmation': False, 'new_method_preference': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Keep the existing acceptance receipt')
    result = {'status': 'paid_pool_stop_accepted', 'cases': [accept(root) for root in args.source],
        'scope': 'read-only development acceptance, not a new scientific comparison',
        'new_compute': 0, 'new_fits': 0, 'new_predictions': 0,
        'new_observations': 0, 'model_calls': 0}
    save(args.output, result)
    print(result)
