#!/usr/bin/env python3
"""Test model-selected recall with and without supported historical methods."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from scripts import run_python_method_reopening_study as prior
from scripts.python_method_edit_practice import EditPracticeMethodLearner

ORIGIN = ROOT / 'artifacts/python_method_edit_practice_luna_20261006'
TRANSFER = ROOT / 'artifacts/python_retained_method_comparison_luna_20261007'
CHOICE_INTERFACE = '''
Continue this unfinished research task using the paid evidence already available.
The method_decision summary lists all available historical programs and their
observed support, failures and untested cases. Decide whether an earlier program
is worth restoring unchanged, should be locally revised, or should remain rejected.
No candidate has been selected for you. You may inspect full files when useful.
Use restore_method or ordinary Python to select a saved source, write to revise,
practice to investigate, and finish to request the existing publication checks.
The host will not choose a historical method after your submission fails.
'''


class AutonomousRecallAuthor(EditPracticeMethodLearner):
    """Experimental ablation of host selection; numerical practice stays shared."""

    def __init__(self, *args, reverse_candidates=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reverse_candidates = reverse_candidates
        self.starting_sources = {}
        self.choice_events = []
        before, marker, tail = self.common_interface.partition('Before these practices, finish')
        _, end, after = tail.partition('Already-confirmed cases')
        if not marker or not end:
            raise ValueError('the study must describe the actual host-selection ablation')
        self.common_interface = before + (
            'Only the source you selected is tested and considered for publication. '
            'A failed case or publication requirement returns its evidence to you; '
            'the host never restores a different source. All original publication '
            'requirements remain in force.\nAlready-confirmed cases') + after + CHOICE_INTERFACE

    def _incumbent_trial(self, current_source):
        return None

    def _prepare_program_state(self):
        super()._prepare_program_state()
        if self.reverse_candidates:
            decision = json.loads(self.files['method_decision.json'])
            decision['candidates'].reverse()
            self.files['method_decision.json'] = json.dumps(decision, indent=2)
            (self.workspace / 'method_decision.json').write_text(self.files['method_decision.json'])

    def _execute(self, action, arguments, laboratory):
        before = (self.workspace / prior.METHOD_PATH).read_text()
        submitted = self.starting_sources.get(before) if action == 'finish' else None
        feedback = super()._execute(action, arguments, laboratory)
        after = (self.workspace / prior.METHOD_PATH).read_text()
        explicit_restore = action == 'run' and arguments.get('operator') == 'restore_method'
        if after != before or explicit_restore or action == 'finish':
            origin = self.starting_sources.get(after)
            self.choice_events.append({'action': action, 'operator': arguments.get('operator'),
                'source_changed': after != before, 'selected_initial_source': origin,
                'selected_supported_source': bool(origin and origin['supported']),
                'novel_edit': after != before and origin is None,
                'submitted_initial_source': submitted,
                'published': bool(feedback.get('published'))})
        return feedback


def source_support(history):
    grouped = {}
    for trial in history['practice_trials']:
        item = grouped.setdefault(trial['method_source'], {'trials': [], 'latest': {}})
        item['trials'].append(trial['directory'])
        item['latest'][trial['case']] = trial
    return {source: {'trials': item['trials'], 'supported': all(
        trial['status'] == 'completed' and (trial.get('fresh_confirmation') or {}).get('passed') is True
        for trial in item['latest'].values())} for source, item in grouped.items()}


def remove_supported_history(history, output):
    """Remove real supported sources; never relabel a success as a failure."""
    ablated = deepcopy(history)
    support = source_support(history)
    kept = [trial for trial in history['practice_trials'] if not support[trial['method_source']]['supported']]
    output.mkdir(parents=True, exist_ok=False)
    trials = []
    for index, trial in enumerate(kept):
        directory = f'trials/trial_{index:03d}'
        shutil.copytree(Path(history['workspace']) / trial['directory'], output / directory)
        trials.append({**deepcopy(trial), 'directory': directory})
    ablated.update(workspace=str(output), practice_trials=trials)
    prior.write_json(output / 'result.json', ablated)
    return ablated


def run_study(output, *, provider_factory=None, auth_mode=None):
    output = Path(output).resolve()
    original_plan = json.loads((ORIGIN / 'protocol.json').read_text())
    history = json.loads((ORIGIN / 'author_2/learning/result.json').read_text())
    sources = json.loads((ORIGIN / 'existing_source_outcomes.json').read_text())['sources']
    jobs = [{'id': f'{condition}_{index:02d}', 'condition': condition, 'index': index,
             'reverse_candidates': bool(index % 2)} for index in range(10)
            for condition in ('supported_memory', 'without_supported_memory')]
    limits = {**original_plan['author_limits'], 'max_model_calls': 8, 'token_budget': 350000,
              'observation_budget': 2400, 'compute_budget': 1500000, 'execution_seconds': 180.}
    plan = {'study': 'model_selected_research_method_recall', 'model': 'gpt-6-luna',
        'provider': {'backend': 'codex_cli', 'reasoning_effort': 'high', 'timeout_seconds': 180,
                     'auth_mode': auth_mode}, 'authors': jobs, 'author_limits': limits,
        'previous_result': str(ORIGIN / 'author_2/learning/result.json'),
        'source_archive_index': original_plan['author_source_index'],
        'practice_cases': original_plan['author_replicas'][2]['practice_cases'],
        'confirmation_streams': 'Add 230003 * replica_index to future confirmation seeds; previously '
            'paid support remains prior evidence. Existing practice-index increments remain.',
        'intervention': 'Show every stored source and its latest paid support in a direct decision '
            'summary. Alternate candidate display order. No recommended or preselected source. '
            'The model alone selects restored sources; host fallback is disabled in both groups.',
        'memory_ablation': 'Remove all supported source records and their local trial directories, '
            'retaining actual failed sources and outcomes. Renumber retained trial directories '
            'to keep the ordinary practice allocator unchanged. Never flip outcome labels.',
        'metrics': 'Count explicit recall or copying, confirmed unchanged recall, novel revisions, '
            'unmodified submissions of initially refuted sources, inspection-only attempts and '
            'service failures separately. Retrieving a refuted source to revise it is not an error.',
        'failure_policy': 'Twenty fixed attempts, no replacement or success-conditioned continuation.',
        'transfer': 'This study measures source selection and the six-case publication gate. For '
            'the exact previously evaluated published source, reference its twelve frozen outcomes '
            'once; no new target execution or transfer replication. Newly written sources have '
            'publication evidence only and are not given additional transfer claims.',
        'scope': 'Repeated model decisions on one historical regression state. Ten runs per '
            'memory condition are not ten independent task families. The change bundles a '
            'direct evidence summary, clearer continuation instructions and an eight-call limit; '
            'comparison with earlier four-call studies does not isolate any one component.'}
    output.mkdir(parents=True, exist_ok=False)
    prior.write_json(output / 'protocol.json', plan)
    for path in (Path(__file__), ROOT / 'src/hegel_machine/cognition/python_method_learning.py',
                 ROOT / 'src/hegel_machine/cognition/code_provider_context.py'):
        shutil.copy2(path, output / (path.stem + '_at_start.py'))
    ablated = remove_supported_history(history, output / 'memory_without_support')
    histories = {'supported_memory': history, 'without_supported_memory': ablated}
    original_source = (ORIGIN / 'unchanged_method.py').read_text()

    def author(job):
        path = output / job['id'] / 'learning'
        provider = provider_factory(path) if provider_factory else prior.CodexCodeProvider(path,
            model=plan['model'], reasoning_effort='high', timeout_seconds=180)
        source = sources[plan['source_archive_index']]
        memory = prior.archive_memory(source)
        memory['files'][prior.METHOD_PATH] = original_source
        cases = deepcopy(plan['practice_cases'])
        for case in cases.values():
            case['confirmation_seed'] += 230003 * job['index']
        learner = AutonomousRecallAuthor(provider, path, knowledge=memory, cases=cases,
            lab_factory=prior.laboratory, config=prior.InquiryLLMConfig(**limits),
            prepare_trial=prior.trial_preparer(source, original_plan), counterexample_case='case_5',
            reverse_candidates=job['reverse_candidates'])
        prior.supply_counterexample(learner, ORIGIN / 'paid_source_counterexample.json', case_label='case_5')
        saved = histories[job['condition']]
        learner.restore_practice(saved)
        learner.starting_sources = source_support(saved)

        def checkpoint(result):
            prior.write_json(path / 'active.json', result)
            event = result['events'][-1] if result['events'] else {}
            print(json.dumps({'author': job['id'], 'call': result['costs']['model_calls'],
                'action': event.get('action'), 'operator': event.get('arguments', {}).get('operator'),
                'restored': event.get('feedback', {}).get('restored_method'),
                'published': bool(event.get('feedback', {}).get('published'))}), flush=True)

        try:
            result = learner.run(on_checkpoint=checkpoint)
        except Exception:
            result = learner.result('execution_failed')
            result['execution_error'] = traceback.format_exc()
        result['model_choices'] = learner.choice_events
        result['current_source_coverage'] = learner._current_source_coverage()
        prior.write_json(path / 'result.json', result)
        final_source = result.get('method_source')
        restored = [choice for choice in learner.choice_events
                    if choice['action'] != 'finish' and choice['source_changed']
                    and choice['selected_initial_source'] is not None]
        successful = result['status'] == 'completed'
        summary = {**job, 'status': result['status'], 'published': successful,
            'autonomous_supported_recall': any(choice['selected_supported_source'] for choice in restored),
            'confirmed_unchanged_recall': successful and bool(
                learner.starting_sources.get(final_source, {}).get('supported')) and any(
                    choice['selected_supported_source'] for choice in restored),
            'recalled_refuted_source': any(not choice['selected_supported_source'] for choice in restored),
            'revised': any(choice['novel_edit'] for choice in learner.choice_events),
            'submitted_initially_refuted_source': any(choice['submitted_initial_source'] is not None
                and not choice['submitted_initial_source']['supported'] for choice in learner.choice_events),
            'inspection_only': not learner.choice_events and not any(
                event.get('arguments', {}).get('operator') == 'practice' for event in result['events']),
            'new_practices': len(result['practice_trials']) - len(saved['practice_trials']),
            'costs': result['costs'], 'usage': prior.usage_summary(result),
            'coverage': result['current_source_coverage']}
        prior.write_json(output / job['id'] / 'summary.json', summary)
        return summary, final_source

    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(author, jobs))
    authors = [item for item, _ in completed]
    reference = json.loads((TRANSFER / 'summary.json').read_text())
    known_source = json.loads((TRANSFER / 'author_2/learning/result.json').read_text())['method_source']
    referenced = [item['id'] for item, source in completed if item['published'] and source == known_source]
    metrics = ('published', 'autonomous_supported_recall', 'confirmed_unchanged_recall',
               'recalled_refuted_source', 'revised', 'submitted_initially_refuted_source', 'inspection_only')
    summary = {'study': plan['study'], 'authors': authors,
        'conditions': {condition: {'attempts': len(rows),
            **{key: sum(row[key] for row in rows) for key in metrics},
            'provider_failed': sum(row['status'] == 'provider_failed' for row in rows)}
            for condition in histories for rows in [[a for a in authors if a['condition'] == condition]]},
        'new_author_costs': prior.total_costs([item['costs'] for item in authors]),
        'author_usage': {key: sum(item['usage'][key] for item in authors) for key in authors[0]['usage']},
        'reused_target_evidence': {'reference': str(TRANSFER / 'summary.json'),
            'authors_with_exact_published_source': referenced,
            'unique_historical_targets': reference['new_target_executions'] if referenced else 0,
            'confirmed_and_correct': reference['target_cohorts']['cohort_0']['confirmed_and_correct'] if referenced else 0},
        'new_target_executions': 0, 'scope': plan['scope']}
    prior.write_json(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'authors'}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_study(args.output, auth_mode=prior.verify_codex_subscription())
