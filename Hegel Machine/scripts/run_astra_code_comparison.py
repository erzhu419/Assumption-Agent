#!/usr/bin/env python3
"""Same-history Astra Code rescue, followed by independent live Code targets."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.run_anonymous_method_learning import laboratory, score_frozen
from scripts.run_anonymous_online_research import prepare_representation_prefix
from scripts.run_anonymous_revision_inquiry import run_study
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def protocol(baseline):
    baseline = Path(baseline).resolve()
    saved = read_json(baseline / 'protocol.json')
    online = deepcopy(saved['online'])
    online.update(study='astra_code_same_paid_history', arms=['code_react'])
    online['scope'] = ('Same pre-Astra paid scene, model, reasoning effort and additional limits; '
        'Code has the common tools and inherited research records without automatic Hegel root focus. '
        'This is a same-history continuation, not a cold-start plain Code agent.')
    return {'study': 'astra_code_source_and_independent_targets', 'baseline': str(baseline),
        'resume_result': saved['resume_result'], 'resume_workspace': saved['resume_workspace'],
        'source_archive': str(Path(saved['source_study']) / 'initial_archive.json'),
        'target_archive': str(Path(saved['method']['source']) / 'initial_archive.json'),
        'online': online, 'method': deepcopy(saved['method']),
        'target_model_limits': {'max_model_calls': 96, 'token_budget': 10_000_000,
                                'execution_seconds': 7200.},
        'target_protocol': ('All 22 unique target settings independently receive only the original '
            'pre-Astra archive and identical 72-row prefixes. Neither the successful rescue nor '
            'the new compiled method nor sibling results transfer. Physical/confirmation limits '
            'and terminal thresholds match each frozen target. Live Code may retry failed '
            'confirmations within its budget, whereas frozen Hegel stopped at its first submission. '
            'This compares live-Code capability with a frozen learned method, not equal-token '
            'causal superiority. All live target research stops before terminal scoring.'),
        'failure_policy': 'Retain completed failures. Resume skips every saved training result; '
            'unfinished live work requires explicit paid-action recovery, never silent replay.'}


def target_limits(plan, case):
    limits = {**plan['method']['limits'], **case.get('limits', {})}
    limits.update(plan['target_model_limits'])
    return InquiryLLMConfig(**limits)


def prepare_target_prefix(agent, lab, case):
    if 'apparatus_seed' in case and case.get('kind') != 'conditional_window':
        return prepare_representation_prefix(agent, lab, case)
    actions = lab.confirmation_actions(72, Random(case['prefix_seed']))
    feedback = agent._execute('query', {'actions': actions}, lab)
    if feedback.get('error'):
        raise ValueError(feedback['error'])
    agent.events.append({'step': -1, 'action': 'query', 'arguments': {'actions': actions},
        'feedback': feedback, 'usage': {'backend': 'common_paid_prefix', 'total_tokens': 0}})


def source_study(plan, output, provider_factory):
    output = Path(output)
    if (output / 'summary.json').exists():
        return read_json(output / 'summary.json')
    previous = read_json(plan['resume_result'])

    def restore(agent, lab, case):
        restored = agent.restore_research(previous, plan['resume_workspace'], lab,
            confirmation_seed=case['confirmation_seed'])
        write_json(output / 'restoration.json', restored)
        write_json(agent.workspace / 'active.json', agent.result('researching'))
        return {'prior_research_cost': deepcopy(previous['costs']),
                'resume_origin': plan['resume_result']}

    return run_study(plan['online'], output, provider_factory=provider_factory,
        knowledge=read_json(plan['source_archive']), prepare_prefix=restore,
        laboratory_factory=laboratory)


def train_target(plan, output, case, knowledge, provider_factory):
    path = Path(output) / 'targets' / case['id'] / 'code_react'
    training = path / 'training.json'
    if training.exists():
        return case, read_json(training)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Unfinished paid target needs recovery before continuation: {path}')
    agent = CommonArchiveInquiryAgent(provider_factory(path), path, arm='code_react',
        knowledge=deepcopy(knowledge), config=target_limits(plan, case), source_task=False,
        confirmation_seed=case['confirmation_seed'])
    lab = laboratory(case)
    prepare_target_prefix(agent, lab, case)
    result = agent.run(lab, on_checkpoint=lambda data: write_json(path / 'active.json', data))
    write_json(training, result)
    return case, result


def run_comparison(plan, output, *, provider_factory, workers=2):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved = output / 'protocol.json'
    if saved.exists():
        if read_json(saved) != plan:
            raise ValueError('Resume requires the same registered comparison protocol')
    else:
        write_json(saved, plan)
    write_json(output / 'progress.json', {'stage': 'same_history_source'})
    source = source_study(plan, output / 'source', provider_factory)
    knowledge = read_json(plan['target_archive'])
    cases = plan['method']['transfer_cases']
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(train_target, plan, output, case, knowledge, provider_factory)
                   for case in cases]
        for future in as_completed(futures):
            case, result = future.result()
            results[case['id']] = (case, result)
            current = {'stage': 'target_research', 'completed': len(results), 'registered': len(cases),
                'last_case': case['id'], 'last_status': result['status'],
                'last_confirmation_passed': bool((result.get('fresh_confirmation') or {}).get('passed'))}
            write_json(output / 'progress.json', current)
            print(json.dumps(current), flush=True)
    write_json(output / 'progress.json', {'stage': 'target_terminal', 'completed': len(results)})
    records = score_frozen(plan['method'], [results[case['id']] for case in cases])
    summary = {'study': plan['study'], 'status': 'completed', 'source': source,
        'targets': records, 'target_terminal_passed': sum(row['terminal']['passed'] for row in records),
        'target_confirmation_passed': sum(row['fresh_confirmation_passed'] for row in records),
        'target_count': len(records), 'scope': plan['target_protocol']}
    write_json(output / 'summary.json', summary)
    write_json(output / 'progress.json', {'stage': 'completed',
        'targets': len(records), 'terminal_passed': summary['target_terminal_passed']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    plan = protocol(args.baseline)
    plan['online']['provider']['auth_mode'] = verify_codex_subscription()
    cfg = plan['online']['provider']
    provider = lambda workspace: CodexCodeProvider(workspace, model=plan['online']['model'],
        reasoning_effort=cfg['reasoning_effort'], timeout_seconds=1800.)
    result = run_comparison(plan, args.output, provider_factory=provider, workers=args.workers)
    print(json.dumps({'targets': result['target_count'],
                      'passed': result['target_terminal_passed']}), flush=True)


if __name__ == '__main__':
    main()
