#!/usr/bin/env python3
"""Expose the actual published entry to both live archive consumers."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.temporal_method_learning import register_published_method
from scripts.run_anonymous_method_learning import laboratory
from scripts.run_anonymous_online_research import prepare_representation_prefix
from scripts.run_anonymous_revision_inquiry import run_study
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json


def run_adoption(source, output, *, provider_factory, auth_mode=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    original = json.loads((source / 'protocol.json').read_text())
    author = json.loads((source / 'author' / 'training.json').read_text())
    memory = register_published_method(
        json.loads((source / 'initial_archive.json').read_text()), author, max_steps=12)
    repair = original.get('study') == 'recurrent_published_method_repair_transfer'
    if repair:
        cases = [deepcopy(case) for case in original['transfer_cases']
                 if case['id'] not in original['old_target_ids']]
    else:
        cases = [deepcopy(case) for case in original['transfer_cases']
                 if case['id'] in original['live_control_cases']]
    recurrent = 'apparatus_seed' in cases[0]
    plan = {'study': 'anonymous_published_method_live_binding',
        'source': str(source), 'arms': ['code_react', 'hegel'], 'cases': cases,
        'limits': original['limits'], 'prefix_rows': original['prefix_rows'],
        'model': original['model'], 'provider': deepcopy(original['provider']),
        'evaluation': original['evaluation'],
        'binding': 'Both arms receive the same original archive, published source, '
            'author entry and paid practice support. The existing procedure execution '
            'action is exposed; the agent chooses whether to invoke it. Ordinary script '
            'execution semantics are unchanged. No new scientific source is provided.',
        'scope': 'Development rerun on the same three targets as the earlier unregistered '
            'archive condition. Independent calls, not independent tasks or a randomized '
            'estimate of interface benefit. Terminal labels are withheld until all research stops.'}
    if auth_mode is not None:
        plan['provider']['auth_mode'] = auth_mode
    if repair:
        plan.update(study='recurrent_repaired_method_live_adoption_development',
            prefix_protocol='Buy 24 independent three-step trajectories without meter readings.',
            binding='Both arms receive the same original K0 archive, newly published method, '
                'author entry and its paid source-practice support. The repair-study archive '
                'remains author-only. The model chooses whether to invoke, inspect or revise '
                'the program through the existing common tools.',
            terminal_order='Every live research episode stops before any terminal scoring.',
            scope=f'Development adoption on all {len(cases)} already-opened new-parameter '
                f'targets of the paired frozen-method study, {2 * len(cases)} Code/Hegel '
                'episodes. Both live arms receive the same revised method. No earlier target '
                'training, frozen outcomes or terminal labels enter research. This measures '
                'online use, not the causal effect of new versus old methods, an independent '
                'task replication, or autonomous recursive self-improvement.')
    elif recurrent:
        plan.update(study='recurrent_published_method_live_continuation_development',
            prefix_protocol=original['prefix_protocol'],
            scope=f'Development rerun on the same {len(cases)} registered recurrent targets, '
                f'{2 * len(cases)} Code/Hegel episodes with independent model calls. '
                'The original archive and published source are unchanged. No prior target '
                'training, frozen-transfer outcomes or terminal labels enter the agents. '
                'These are not independent tasks or a randomized estimate of interface benefit. '
                'Terminal labels are withheld until every research episode stops.')
    plan['execution_workers'] = original.get('execution_workers', 1)
    plan['runtime_snapshot'] = sorted(set(original.get('runtime_snapshot', [])) | {
        'scripts/run_anonymous_method_adoption.py'})
    if (output / 'protocol.json').exists():
        raise ValueError('use a fresh output directory to retain earlier developmental failures')
    for name in plan['runtime_snapshot']:
        destination = output / 'source_at_run' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    summary = run_study(plan, output, provider_factory=provider_factory, knowledge=memory,
        laboratory_factory=laboratory,
        prepare_prefix=prepare_representation_prefix if recurrent else None,
        workers=plan['execution_workers'])
    for trial in summary['trials']:
        training = json.loads((output / trial['case'] / trial['arm'] / 'training.json').read_text())
        method = author['method_path']
        calls = [event for event in training['events']
                 if event.get('arguments', {}).get('path') == method]
        trial['entry_attempts'] = sum(
            event['action'] == 'query' or (event['action'] == 'run' and
                event['arguments'].get('operator') == 'procedure') for event in calls)
        unchanged = [step for step in training['method_results']
            if step['path'] == method and step.get('proposal') is not None
            and step['feedback'].get('inherited_source_unchanged') is True]
        trial['unchanged_published_steps'] = len(unchanged)
        trial['published_method_confirmed'] = any(
            (step['feedback'].get('fresh_confirmation') or {}).get('passed') is True
            for step in unchanged)
        trial['module_only_runs'] = sum(event['action'] == 'run' and
            'operator' not in event['arguments'] for event in calls)
    summary['comparison_scope'] = plan['scope']
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.source / 'protocol.json').read_text())
    summary = run_adoption(args.source, args.output,
        auth_mode=verify_codex_subscription(), provider_factory=lambda workspace:
            CodexCodeProvider(workspace, model=plan['model'], reasoning_effort='medium',
                timeout_seconds=plan['provider']['timeout_seconds']))
    print(json.dumps({'completed': len(summary['trials']),
        'adopting_episodes': sum(row['unchanged_published_steps'] > 0
                                 for row in summary['trials'])}), flush=True)


if __name__ == '__main__':
    main()
