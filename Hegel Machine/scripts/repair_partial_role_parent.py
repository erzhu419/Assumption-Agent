#!/usr/bin/env python3
"""Accept the actual pre-terminal candidate using its retained paid evidence.

This resumes only the lost candidate decision, not its already completed fits.
The original unresolved run and failed terminal remain immutable.
"""
import argparse
from math import isclose
from pathlib import Path
from time import monotonic, process_time

from scripts.run_partial_role_parent import load, save, LIMITS
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.unified_research import compare_resumed_programs


def rows(values):
    return [Observation(row['features'], row['target'], tuple(row.get('history', ())))
            for row in values]


def repair(root):
    started, cpu_started = monotonic(), process_time()
    output = root / 'paid_retention_repair'
    output.mkdir(exist_ok=False)
    report_path = root / 'with_frozen_source/response_report.json'
    report = load(report_path)
    index, decision = next((i, action) for i, action in enumerate(report['actions'])
        if action.get('action') == 'compare_object_continuation_candidates')
    counterexample = report['actions'][index + 1]
    assert counterexample['action'] == 'retain_inherited_relation_counterexample'
    evidence = rows(report['object_continuation_state']['inherited_relation_evidence'][
        :counterexample['observation_rows']])
    paid_before, confirmation = [], None
    for batch in report['new_batches']:
        if batch['probe']['probe_id'] == decision['probe_id']:
            assert batch['channels'] == decision['channels'] == decision['baseline_channels']
            confirmation = rows(batch['observations'])
            break
        paid_before.extend(rows(batch['observations']))
    assert confirmation and all(row in paid_before for row in evidence)
    assert len(confirmation) == decision['observation_rows']
    old, current = (Program.from_dict(decision[key]) for key in ('baseline_program', 'program'))
    old_fallback, new_fallback = decision['baseline_fallback'], decision['fallback']
    budget = SearchBudget(LIMITS['compute_cost'], spent=report['costs']['compute_cost'])
    initial_spent = budget.spent
    save(output / 'registration.json', {
        'report_ref': {'file': str(report_path)}, 'action_index': index,
        'evidence_rows': len(evidence), 'confirmation_probe': decision['probe_id'],
        'remaining_original_compute': budget.remaining,
        'operation': 'same two stored programs; paid decision and confirmation only',
        'new_fits': 0, 'new_observations': 0})
    assert budget.spend(2 * len(confirmation))
    old_loss = loss(old, confirmation, old_fallback, .5)
    new_loss = loss(current, confirmation, new_fallback, .5)
    comparison = compare_resumed_programs(old, current,
        before_loss=old_loss, after_loss=new_loss,
        incumbent_fallback=old_fallback, current_fallback=new_fallback,
        directed_command_rows=[], inherited_relation_evidence=evidence,
        feature_names=(), budget=budget, minimum_improvement=1e-7)
    # Matching the original score also verifies the retained pre-decision slice.
    assert isclose(comparison['checks']['incumbent']['inherited_relation']['loss'],
                   counterexample['loss'], rel_tol=1e-12, abs_tol=1e-25)
    assert isclose(old_loss, decision['before_loss'], rel_tol=1e-12, abs_tol=1e-30)
    assert isclose(new_loss, decision['after_loss'], rel_tol=1e-12, abs_tol=1e-30)
    accepted = (comparison['status'] == 'compared'
                and comparison['retain_incumbent'] is False
                and comparison['checks']['current']['inherited_relation']['loss'] <= 1e-7
                and new_loss <= 1e-7)
    receipt = {
        'repair_receipt_type': 'paid_parent_candidate_retention',
        'source_parent_root': str(root), 'passed': accepted,
        'selected_candidate_origin': {'file': str(report_path),
            'json_path': ['actions', index, 'program']},
        'paid_evidence_rows': len(evidence), 'comparison': comparison,
        'evaluation': {'kind': 'paid_predecision_confirmation', 'rows': len(confirmation),
            'incumbent_mse': old_loss, 'current_mse': new_loss,
            'threshold': 1e-7, 'passed': new_loss <= 1e-7},
        'compute_cost': budget.spent - initial_spent,
        'remaining_original_compute': budget.remaining,
        'observation_cost': 0, 'measurement_cost': 0, 'fits': 0, 'llm_calls': 0,
        'original_native_status': report['status'],
        'original_terminal_ref': {'file': str(root / 'terminal/summary.json')},
        'terminal_reevaluated': False,
        'scope': 'development paid-decision repair; no new independent confirmation',
        'elapsed_seconds': monotonic() - started, 'process_cpu_seconds': process_time() - cpu_started}
    save(output / 'receipt.json', receipt)
    if accepted:
        save(output / 'accepted_predictor.json', {'program': decision['program'],
            'fallback': new_fallback, 'selected_channels': decision['channels'],
            'status': 'accepted_on_retained_paid_confirmation',
            'source_ref': receipt['selected_candidate_origin']})
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(repair(parser.parse_args().root))
