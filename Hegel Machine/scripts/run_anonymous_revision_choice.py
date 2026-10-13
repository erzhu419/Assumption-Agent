#!/usr/bin/env python3
"""Test autonomous refitting or representation revision with one learned method."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import register_published_method
from scripts.run_anonymous_online_research import (
    prepare_representation_prefix, representation_laboratory, research_trace)
from scripts.run_anonymous_revision_inquiry import run_study
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json


def protocol(source):
    source = Path(source).resolve()
    original = json.loads((source / 'protocol.json').read_text())
    cases = []
    for block, seed in enumerate((71231031, 71232040)):
        modes = ('recurrent', 'finite2') if block == 0 else ('finite2', 'recurrent')
        for mode in modes:
            cases.append({'id': f'case_{len(cases):03d}', 'parameter_pair': block,
                'kind': mode, 'seed': seed, 'apparatus_seed': seed,
                'prefix_seed': seed + 100003, 'confirmation_seed': seed + 200003,
                'evaluation_seed': seed + 300007})
    snapshot = set(original.get('runtime_snapshot', [])) | {
        str(path.relative_to(ROOT)) for path in (ROOT / 'src/hegel_machine').rglob('*.py')}
    snapshot.update({'scripts/run_anonymous_revision_choice.py',
        'scripts/run_anonymous_online_research.py', 'scripts/run_anonymous_revision_inquiry.py',
        'scripts/run_anonymous_method_revision.py', 'scripts/run_online_inquiry_llm.py'})
    return {'study': 'anonymous_learned_method_revision_choice', 'source': str(source),
        'model': 'gpt-6-luna', 'provider': {'backend': 'codex_cli',
            'reasoning_effort': 'medium', 'timeout_seconds': 600.},
        'arms': ['code_react', 'hegel'], 'cases': cases, 'execution_workers': 2,
        'limits': asdict(InquiryLLMConfig(max_model_calls=96, token_budget=10_000_000,
            execution_seconds=3600., observation_budget=1152, confirmation_rows=96,
            compute_budget=250_000)),
        'prefix_rows': 72,
        'prefix_protocol': 'Buy 24 independent three-step trajectories without meter readings. '
            'Each parameter pair shares controls and apparatus coefficients. The paired worlds '
            'agree on the short prefix; their public 24-step horizon permits distinguishing '
            'experiments. World labels and task seeds do not enter the researcher context.',
        'binding': 'Both arms receive the same complete original source archive and unchanged '
            'published program with its actual source practice support. No earlier target '
            'observations, fitted target models or terminal outcomes are transferred. The '
            'researcher may invoke, inspect, revise or decline the program.',
        'discovery': 'Reuse ordinary Python, compare_experiments and prospective query revision. '
            'The researcher authors every scientific candidate and proposed experiment. No '
            'fitted alternative, mechanism label, supplied probe or automatic retry is added. '
            'Direct task success counts even without a comparison or revision interface.',
        'evaluation': {'rows': 512, 'trajectory_length': 24, 'mse_threshold': 1e-10},
        'terminal_order': 'All eight live investigations stop before any terminal evaluation. '
            'Terminal outcomes never return to research.',
        'failure_policy': 'Keep every registered episode and actual failure. Limits are identical '
            'and fixed before research; no performance-dependent retries or additional allocation.',
        'outcome_protocol': 'Report task success and costs separately from actual method execution, '
            'method source changes, prospective comparisons, acquisitions and subsequent decisions. '
            'A declaration, a method stop, or unequal forecasts alone is not successful discrimination.',
        'scope': 'Two fresh parameter blocks, two historical apparatus families and eight live '
            'Code/Hegel episodes using one previously learned method. This tests autonomous '
            'refit-versus-revision behavior in a limited development setting. It does not isolate '
            'recursive organization, establish open-world discovery or replace the preserved '
            'six-target frozen-transfer evidence.',
        'runtime_snapshot': sorted(snapshot)}


def record_research(summary, output, author):
    """Summarize actual executions without deciding the scientific explanation."""
    output = Path(output)
    method = author['method_path']
    for trial in summary['trials']:
        workspace = output / trial['case'] / trial['arm']
        training = json.loads((workspace / 'training.json').read_text())
        trace = research_trace(training)
        write_json(workspace / 'research_trace.json', trace)
        calls = [event for event in training['events']
                 if event.get('arguments', {}).get('path') == method]
        steps = [step for step in training['method_results'] if step['path'] == method]
        executed = [step for step in steps if step.get('proposal') is not None]
        comparisons = training.get('experiment_comparisons', [])
        revisions = training.get('revision_proposals', [])
        trial.update(
            published_method_entry_attempts=sum(event['action'] == 'query' or
                (event['action'] == 'run' and event['arguments'].get('operator') == 'procedure')
                for event in calls),
            published_method_executed_steps=len(executed),
            published_method_stops=sum(isinstance(step.get('proposal'), dict) and
                'stop' in step['proposal'] for step in executed),
            unchanged_published_steps=sum(step['feedback'].get('inherited_source_unchanged') is True
                for step in executed),
            published_method_confirmed=any(
                step['feedback'].get('inherited_source_unchanged') is True and
                (step['feedback'].get('fresh_confirmation') or {}).get('passed') is True
                for step in executed),
            published_method_present_at_end=(workspace / method).is_file(),
            published_method_changed_at_end=((workspace / method).read_text() != author['method_source']
                if (workspace / method).is_file() else None),
            other_executed_method_paths=sorted({step['path'] for step in training['method_results']
                if step['path'] != method and step.get('proposal') is not None}),
            experiment_comparison_attempts=len(comparisons),
            completed_experiment_comparisons=sum(row.get('status') == 'compared' for row in comparisons),
            prospective_revision_attempts=len(revisions),
            acquired_prospective_revisions=sum(bool(row.get('observations')) for row in revisions),
            prospective_revisions_with_different_forecasts=sum(
                row['predictions_differ'] and row['acquisition_step'] is not None
                for row in trace['prospective_revisions']),
            research_trace=str(workspace / 'research_trace.json'))
    summary['research_interpretation'] = (
        'Counts describe actual recorded interface use, not inferred scientific success. '
        'Inspect frozen forecasts, paid outcomes, authored readouts and subsequent actions '
        'in research_trace.json and training.json. An unchanged published source may coexist '
        'with a separately authored revised predictor or method.')
    write_json(output / 'summary.json', summary)
    return summary


def run_choice(source, output, *, provider_factory, auth_mode=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    plan = protocol(source)
    if auth_mode is not None:
        plan['provider']['auth_mode'] = auth_mode
    author = json.loads((source / 'author/training.json').read_text())
    memory = register_published_method(
        json.loads((source / 'initial_archive.json').read_text()), author, max_steps=12)
    if (output / 'protocol.json').exists():
        raise ValueError('use a fresh output directory to preserve all earlier investigations')
    for name in plan['runtime_snapshot']:
        destination = output / 'source_at_run' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    summary = run_study(plan, output, provider_factory=provider_factory, knowledge=memory,
        laboratory_factory=representation_laboratory, prepare_prefix=prepare_representation_prefix,
        workers=plan['execution_workers'])
    return record_research(summary, output, author)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = run_choice(args.source, args.output, auth_mode=verify_codex_subscription(),
        provider_factory=lambda workspace: CodexCodeProvider(workspace, model='gpt-6-luna',
            reasoning_effort='medium', timeout_seconds=600.))
    print(json.dumps({'completed': len(summary['trials']),
        'terminal_passed': sum(row['terminal']['passed'] for row in summary['trials'])}), flush=True)


if __name__ == '__main__':
    main()
