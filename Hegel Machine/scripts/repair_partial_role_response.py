#!/usr/bin/env python3
"""Consume a paid child answer before buying one newly frozen confirmation.

One registered development unit on the first failed PR3 task; the original
cohort, its complete pool, terminal observations and bills remain unchanged.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
from time import monotonic, process_time

from run_partial_role_parent import load, save
from run_partial_role_successors import historical_knowledge
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry, observations, restore_object_state
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


LIMIT = 90000
SEED = 106029101
CONFIRMATION_ROWS = 48
CONFIRMATION_SEED = SEED + 831271


def repair(root, acceptance, output):
    started, cpu = monotonic(), process_time()
    registration = load(root / 'registration.json')
    report_path = root / 'runs' / str(SEED) / 'pr3' / 'response_report.json'
    report = load(report_path)
    seed_receipt = load(acceptance / 'receipt.json')
    # The acceptance script supplies its complete bill including its retained
    # unsuccessful restore; this is one capped unit, not a renewed allowance.
    seed_compute = seed_receipt['total_compute_cost']
    budget = SearchBudget(LIMIT, spent=seed_compute)
    output.mkdir(parents=True, exist_ok=False)
    settings = next(item for item in registration['tasks'] if item['seed'] == SEED)
    save(output / 'registration.json', {
        'scope': 'paid parent-answer repair; not replacement ABC results',
        'report_ref': str(report_path), 'seed_acceptance_ref': str(acceptance / 'receipt.json'),
        'historical_knowledge_ref': registration['historical_knowledge_ref'],
        'original_costs': report['costs'], 'development_compute_limit': LIMIT,
        'already_spent_development_compute': seed_compute,
        'maximum_new_confirmation_rows': CONFIRMATION_ROWS,
        'confirmation_seed': CONFIRMATION_SEED, 'confirmation_threshold': 1e-7,
        'rule': 'consume the first stored structural answer; compare full retained pool on paid rows; '
                'one fresh confirmation only after improvement; no new fit or terminal reuse'})
    knowledge, _, _ = historical_knowledge(registration['historical_knowledge_ref'])
    agent = UnifiedResearchAgent(ResearchConfig(seed=SEED, compute_budget=LIMIT,
        minimum_improvement=1e-7), library=KnowledgeLibrary.from_dict(knowledge))
    state = restore_object_state(agent, report, ('fine',))
    parent = decode_entry(state['selected'])
    parent['fallback'] = report['fallback']
    pool = [decode_entry(item) for item in state['finalists']]
    for frontier in state['candidate_frontiers']:
        pool.extend(decode_entry(item) for item in frontier['entries'])
    # Newly executed seed candidates are retained too; none are fitted again.
    seed_claims = load(acceptance / 'new_claims.json')
    pool.extend(decode_entry(item) for item in seed_claims)
    support_index, support = next((i, event['noise_assessment'])
        for i, event in enumerate(report['actions'])
        if (event.get('noise_assessment') or {}).get('structural_counterexample'))
    channels = tuple(report['selected_channels'])
    paid = [row for batch in report['batches'] if tuple(batch['channels']) == channels
            for row in observations(batch['observations'])]
    actions, graph = [], ResearchGraph()
    before = budget.spent
    confirmation = None
    with AnonymousObjectLab(**settings) as laboratory:
        spec = agent._spec(laboratory, channels)
        # Score both frozen candidates and check all paid counterexamples.
        close_quote = 2 * CONFIRMATION_ROWS
        budget.reserve = close_quote
        outcome = agent._consume_residual_answer(spec, support, paid, budget,
            actions, graph, parent=parent, finalists=pool)
        budget.release_reserve()
        selected = outcome['selected']
        save(output / 'working_selection.json', {
            'source_support_action': support_index, 'status': outcome['status'],
            'action': outcome['action'], 'paid_rows': len(paid),
            'retained_pool_ref': {'report': str(report_path), 'state': 'object_continuation_state'},
            'seed_claims_ref': str(acceptance / 'new_claims.json'),
            'scores': [{'candidate_id': item['candidate'].candidate_id,
                        'loss': item['validation_loss']} for item in outcome.get('compared_claims', [])]})
        if outcome['status'] == 'improved' and budget.can_spend(close_quote):
            # Persist the actual frozen answer before the new physical query.
            save(output / 'selected_entry.json', encode_entry(selected))
            save(output / 'frozen_predictor.json', {
                'program': selected['candidate'].program.to_dict(),
                'fallback': selected['fallback'], 'selected_channels': list(channels),
                'status': 'selected_on_paid_rows_requires_confirmation',
                'selected_entry_ref': 'selected_entry.json',
                'consumed_support_ref': {'file': str(report_path), 'action': support_index}})
            probe = laboratory.random_probe(CONFIRMATION_SEED, length=CONFIRMATION_ROWS,
                name='paid_parent_answer_repair_confirmation')
            batch = laboratory.acquire(probe, channels)
            save(output / 'confirmation_observations.json', {
                'probe': asdict(probe), 'batch': asdict(batch)})
            budget.spend(close_quote)
            old_loss = loss(parent['candidate'].program, batch.observations, parent['fallback'], .5)
            new_loss = loss(selected['candidate'].program, batch.observations, selected['fallback'], .5)
            confirmation = {'rows': len(batch.observations), 'incumbent_mse': old_loss,
                'selected_mse': new_loss, 'threshold': 1e-7,
                'passed': new_loss <= 1e-7 and selected['validation_loss'] <= 1e-7,
                'fresh_prediction_pass': new_loss <= 1e-7,
                'paid_evidence_pass': selected['validation_loss'] <= 1e-7}
        costs = dict(laboratory.costs)
    receipt = {'status': 'confirmed' if confirmation and confirmation['passed'] else 'unresolved',
        'selection_status': outcome['status'], 'selection': outcome['action'],
        'original_report_ref': str(report_path), 'original_status': report['status'],
        'original_costs': report['costs'], 'paid_rows': len(paid),
        'confirmation': confirmation, 'compute_limit': LIMIT,
        'seed_acceptance_compute': seed_compute, 'answer_compute': budget.spent - before,
        'total_compute_cost': budget.spent, **costs,
        'additional_fits': 0, 'llm_calls': 0,
        'seconds': monotonic() - started, 'python_seconds': process_time() - cpu,
        'scope': 'one development repair; independent method utility and ABC contrast untested'}
    save(output / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(repair(args.root, args.acceptance, args.output))
