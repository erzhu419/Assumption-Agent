#!/usr/bin/env python3
"""Revise a published method from one paid development counterexample.

Source artifacts stay read only. All target slots are fixed before authoring;
both frozen methods finish research before either receives terminal evaluation.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.anonymous_method_learning import AnonymousResearchMethodLearner
from hegel_machine.cognition.code_agent import CodexCodeProvider, python_model_context_matches
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH, register_published_method
from scripts.run_anonymous_method_learning import laboratory, trial_preparer, frozen_transfers, score_frozen
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_python_method_learning_study import usage_summary
from scripts.run_temporal_counterexample_learning import supply_counterexample


COUNTEREXAMPLE_CASE = 'case_006'
COUNTEREXAMPLE = {'kind': 'hysteresis', 'seed': 62113038,
    'prefix_seed': 62213552, 'confirmation_seed': 62313561}


def protocol(source, *, model='gpt-6-luna', timeout_seconds=300., continue_author=None):
    source = Path(source).resolve()
    if continue_author is not None:
        path = Path(continue_author).resolve()
        previous = json.loads(path.read_text())
        if previous.get('method_source') is not None or previous['status'] not in {
                'provider_failed', 'token_budget_exhausted', 'model_call_budget_exhausted'}:
            raise ValueError('continue an actual unpublished interrupted or exhausted author')
        plan = json.loads((path.parent.parent / 'protocol.json').read_text())
        if Path(plan['source']).resolve() != source:
            raise ValueError('continuation must retain the same published source archive')
        earlier = json.loads((path.parent.parent / 'summary.json').read_text())
        plan.update(continued_author=str(path), model=model,
            prior_author_phases=[*earlier['prior_author_phases'], {
                'result': str(path), 'status': previous['status'],
                'costs': previous['costs'], 'usage': usage_summary(previous)}],
            continuation='Additional author allowance with the same seven practices and fixed targets. '
                'Restore every paid draft and failure. Reuse the original paid counterexample without '
                'reacquisition; connect the historical counterexample ledger and revision task. '
                'No fitted source, scientific diagnosis or repair is supplied by the host.')
        plan['provider'] = {**plan['provider'], 'timeout_seconds': timeout_seconds}
        return plan
    old = json.loads((source / 'protocol.json').read_text())
    author = json.loads((source / 'author' / 'training.json').read_text())
    if author['status'] != 'completed' or not author.get('method_source'):
        raise ValueError('revision requires an actually published source method')
    practice = deepcopy(old['practice_cases'])
    if COUNTEREXAMPLE_CASE in practice:
        raise ValueError('this development counterexample is already registered')
    practice[COUNTEREXAMPLE_CASE] = deepcopy(COUNTEREXAMPLE)
    targets = deepcopy(old['transfer_cases'])
    first_by_family = {}
    for case in old['transfer_cases']:
        first_by_family.setdefault(case['kind'], case)
    for index, original in enumerate(first_by_family.values()):
        seed = original['seed']
        for prefix in range(8):
            targets.append({'id': f'new_object_{index:03d}_prefix_{prefix:03d}',
                'kind': original['kind'], 'seed': seed + 3100033,
                'prefix_seed': seed + 4100039 + prefix,
                'confirmation_seed': seed + 5100077 + prefix,
                'evaluation_seed': seed + 6100091 + prefix})
    return {'study': 'anonymous_published_method_counterexample_revision',
        'source': str(source), 'model': model,
        'provider': {'backend': 'codex_cli', 'timeout_seconds': timeout_seconds,
                     'reasoning_effort': 'medium'},
        'author_limits': {**old['author_limits'], 'max_model_calls': 24,
                          'token_budget': 1500000, 'observation_budget': 6000},
        'limits': deepcopy(old['limits']), 'prefix_rows': old['prefix_rows'],
        'practice_cases': practice, 'transfer_cases': targets,
        'old_target_ids': [case['id'] for case in old['transfer_cases']],
        'evaluation': deepcopy(old['evaluation']),
        'counterexample': {'case': COUNTEREXAMPLE_CASE, **deepcopy(COUNTEREXAMPLE),
            'origin': 'Development failure found at the sixth ordinary random prefix on the '
                'previously practiced object. Selected after observing that failure; not held out.',
            'confirmation_rule': 'The first reproduction uses this exact confirmation seed. '
                'Later ordinary practices retain the existing seed plus practice-count rule.'},
        'restoration': 'Restore all earlier paid source drafts and practice results without replay. '
            'The published program is a revisable starting point, not publication on the new case.',
        'comparison': 'Same original executable archive and limits for both methods. Retain the old '
            'nine targets and add three new objects with eight preregistered ordinary prefixes each. '
            'No live consumer or externally selected target repairs; each frozen program acts itself.',
        'terminal_order': 'Finish both methods on all target slots before generating terminal labels.',
        'failure_policy': 'Retain paid failed authoring and reproduction. Without publication, all '
            'paired target slots remain unavailable; do not substitute an unpublished draft.',
        'scope': 'One development counterexample and one author continuation. Prefix repetitions '
            'on each object are not independent discoveries; the retained old targets are revisits.'}


def reproduce_counterexample(author):
    """Execute the exact earlier failing condition and record it on the author account."""
    case = author.cases[COUNTEREXAMPLE_CASE]
    source = (author.workspace / METHOD_PATH).read_text()
    directory = f'trials/trial_{len(author.practice_trials):03d}'
    limits = replace(author.practice_limits, max_model_calls=0, token_budget=0)
    payload = author.prepare_trial(case, author.workspace / directory, source, limits)
    result = payload['agent'].execute_method(payload['laboratory'], max_steps=12)
    selected_prior = bool(result.get('model') and any(
        python_model_context_matches(payload['agent'].workspace, result['model'],
            candidate['snapshot_workspace'], candidate['inner_model'])
        for candidate in payload['inherited_candidates']))
    feedback = author._record_practice(COUNTEREXAMPLE_CASE, directory, source,
        payload['agent'], result, selected_prior)
    author.events.append({'step': -1, 'action': 'run',
        'arguments': {'operator': 'practice', 'case': COUNTEREXAMPLE_CASE},
        'feedback': deepcopy(feedback),
        'usage': {'backend': 'development_counterexample_reproduction', 'total_tokens': 0}})
    return {**feedback, 'effective_confirmation_seed': case['confirmation_seed']}


def attach_revision_task(author, result):
    """Reuse the historical counterexample view; the author chooses the repair."""
    task = (
        'Revise the existing research program using its paid counterexample.\n'
        'The current draft may be unsupported. Earlier sources and every paid practice\n'
        'remain available in method_candidates.json and resume_context.json.\n'
        'counterexample.json identifies the actual rejected execution on ' + COUNTEREXAMPLE_CASE +
        ',\nits executed source, proposals and feedback. Inspect that evidence and decide\n'
        'which source to retain or revise, preserving behavior still supported by the\n'
        'other supplied cases. Use practice and finish to establish support for the\n'
        'final source on every listed case. Only the published self-contained method\n'
        'and original archive transfer. The earlier archive is background evidence.\n\n')
    # Adapt the old reopening task, keeping the existing author actions, paid
    # practice and publication requirements rather than replacing their runtime.
    author.common_interface = task + 'AUTHOR ACTIONS:' + author.common_interface.split('AUTHOR ACTIONS:', 1)[1]
    author.common_interface = author.common_interface.replace(
        'The initial method is only an unresolved placeholder. You author its decisions,\n'
        'predictors, parameter estimation and experiments using ordinary Python.',
        'The inherited method and saved drafts are revisable Python programs.\n'
        'Their paid outcomes identify existing support and unresolved failures.')
    return supply_counterexample(author, result, case_label=COUNTEREXAMPLE_CASE)


def run_revision(plan, output, *, provider_factory):
    output, source = Path(output).resolve(), Path(plan['source'])
    if (output / 'protocol.json').exists():
        raise ValueError('use a fresh output directory; retain earlier author attempts')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', plan)
    old_plan = json.loads((source / 'protocol.json').read_text())
    previous = json.loads((source / 'author' / 'training.json').read_text())
    original_knowledge = json.loads((source / 'initial_archive.json').read_text())
    sources = json.loads((source / 'source_outcomes.json').read_text())
    knowledge = register_published_method(original_knowledge, previous, max_steps=12)
    write_json(output / 'initial_archive.json', knowledge)
    write_json(output / 'source_outcomes.json', sources)
    reference = output / 'reference_archive'
    shutil.copytree(source / 'reference_archive', reference)
    workspace = output / 'author'
    author = AnonymousResearchMethodLearner(provider_factory(workspace), workspace,
        knowledge=knowledge, cases=plan['practice_cases'], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan['author_limits']),
        practice_limits=InquiryLLMConfig(**plan['limits']),
        prepare_trial=trial_preparer(knowledge, sources, reference))
    restored = previous
    if plan.get('continued_author'):
        path = Path(plan['continued_author'])
        restored = json.loads(path.read_text())
        baseline = json.loads((path.parent.parent / 'baseline_reproduction.json').read_text())
    author.restore_practice(restored)
    if not plan.get('continued_author'):
        baseline = reproduce_counterexample(author)
    counterexample = json.loads((workspace / baseline['directory'] / 'result.json').read_text())
    attach_revision_task(author, counterexample)
    write_json(output / 'baseline_reproduction.json', baseline)
    write_json(workspace / 'active.json', author.result('researching'))
    learned = author.run(on_checkpoint=lambda data: write_json(workspace / 'active.json', data))
    write_json(workspace / 'training.json', learned)
    published = learned['status'] == 'completed' and learned.get('method_source') is not None
    write_json(output / 'published_method.json', {'published': published,
        'source': learned.get('method_source'), 'lesson': learned.get('lesson'),
        'practice_trials': learned['practice_trials']})
    frozen = {'original': [], 'revised': []}
    if published:
        for name, publication in (('original', previous), ('revised', learned)):
            memory = register_published_method(original_knowledge, publication, max_steps=12)
            branch = output / name
            branch.mkdir()
            write_json(branch / 'transfer_archive.json', memory)
            frozen[name] = frozen_transfers(plan, branch, memory, sources, publication['method_source'])
    # No earlier scoring: both frozen research cohorts have now stopped.
    records = {name: score_frozen(plan, results) for name, results in frozen.items()}
    pairs = []
    for old, new in zip(records['original'], records['revised']):
        pairs.append({'case': old['case'], 'kind': old['kind'],
            'group': 'old_target' if old['case'] in plan['old_target_ids'] else 'new_object_prefix',
            'original_passed': old['terminal']['passed'], 'revised_passed': new['terminal']['passed'],
            'original_status': old['status'], 'revised_status': new['status'],
            'observation_delta': new['costs']['observation_cost'] - old['costs']['observation_cost'],
            'compute_delta': new['costs']['compute_cost'] - old['costs']['compute_cost']})
    target_costs = {name: total_costs([result['costs'] for _, result in results])
                    for name, results in frozen.items()}
    earlier_phases = plan.get('prior_author_phases') or [*old_plan.get('prior_author_phases', []), {
        'result': str(source / 'author' / 'training.json'), 'status': previous['status'],
        'costs': previous['costs'], 'usage': usage_summary(previous)}]
    summary = {'study': plan['study'], 'status': 'completed', 'method_published': published,
        'source_changed': published and learned['method_source'] != previous['method_source'],
        'author_status': learned['status'], 'author_usage': usage_summary(learned),
        'prior_author_phases': earlier_phases,
        'old_author_costs': total_costs([phase['costs'] for phase in earlier_phases]),
        'baseline_costs': baseline['costs'], 'author_costs': learned['costs'],
        'baseline_cost_scope': ('Already included in prior_author_phases; no new acquisition or charge.'
            if plan.get('continued_author') else
            'Included in author_costs and new_costs exactly once.'),
        'baseline_reused': bool(plan.get('continued_author')),
        'target_costs': target_costs,
        'new_costs': total_costs([learned['costs'], *target_costs.values()]),
        'restored_practice_trials': len(restored['practice_trials']),
        'new_practice_trials': len(learned['practice_trials']) - len(restored['practice_trials']),
        'frozen': records, 'pairs': pairs,
        'unavailable_target_episodes': 0 if published else 2 * len(plan['transfer_cases']),
        'live_consumer_model_calls': 0, 'scope': plan['scope']}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--timeout-seconds', type=float, default=300.)
    parser.add_argument('--continue-author', type=Path,
        help='Restore a real unpublished author phase and its paid counterexample without replay')
    args = parser.parse_args()
    plan = protocol(args.source, model=args.model, timeout_seconds=args.timeout_seconds,
                    continue_author=args.continue_author)
    plan['provider']['auth_mode'] = verify_codex_subscription()
    def provider(workspace):
        return CodexCodeProvider(workspace, model=args.model, reasoning_effort='medium',
            timeout_seconds=args.timeout_seconds)
    summary = run_revision(plan, args.output, provider_factory=provider)
    print(json.dumps({'method_published': summary['method_published'],
        'source_changed': summary['source_changed'], 'new_costs': summary['new_costs']}), flush=True)


if __name__ == '__main__':
    main()
