#!/usr/bin/env python3
"""Test transfer of observation research experience conditioned on executable revision."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import run_observation_action_study as common

ROOT = common.ROOT
OUTPUT = ROOT / 'artifacts/observation_applicability_study_20260919'
ARMS = ('baseline_v093', 'learned', 'policy_disabled')
common.ARMS = ARMS


def prepare(output):
    if (output / 'protocol.json').exists():
        return
    prior = ROOT / 'artifacts/observation_action_study_20260919'
    protocol = json.loads((prior / 'protocol.json').read_text())
    baseline = prior / 'runtime/src'
    if not (baseline / 'hegel_machine/cognition/unified_session.py').exists():
        raise FileNotFoundError(baseline)
    protocol.update(
        experiment='observation_method_applicability_from_prepared_control_revision',
        fixed_at=datetime.now(timezone.utc).isoformat(),
        cohorts={'development': [95201900], 'main': [95201911, 95201912, 95201913, 95201914]},
        instrument_modes={'95201900': 'recurrent', '95201911': 'finite2', '95201912': 'finite2',
                          '95201913': 'recurrent', '95201914': 'recurrent'},
        arms={'baseline_v093': {'runtime': str(baseline.relative_to(ROOT))},
              'learned': {'runtime': str((output / 'runtime/src').relative_to(ROOT))},
              'policy_disabled': {'runtime': str((output / 'runtime/src').relative_to(ROOT)),
                                  'research_config_overrides': {'learn_research_policy': False}}},
        scope='The unchanged seven-stage stream tests whether actual prepared control revisions '
              'condition transfer of observation research experience. Four new apparatus roots, '
              'two finite-prefix and two recurrent-prefix, start from empty libraries; source costs '
              'and failed research are included. No supplied source program, truth parameter, '
              'diagnostic action, candidate score or required measurement. The known recurrent '
              'development stream is separate. All cohort learning precedes terminal evaluation.',
        capability_endpoints=['prepared_control_revision_applicability', 'json_context_specific_experience',
                              'paid_observation_action_choices', 'fresh_response_confirmation',
                              'instrument_and_mechanism_recovery', 'unnecessary_remeasurement'],
        baseline_protocol=str((prior / 'protocol.json').relative_to(ROOT)))
    common.save(output / 'protocol.json', protocol)


def load(protocol, arm, cohort):
    source = ROOT / (protocol['development_runtime'] if cohort == 'development' and arm != 'baseline_v093'
                     else protocol['arms'][arm]['runtime'])
    sys.path.insert(0, str(source))
    session = importlib.import_module('hegel_machine.cognition.unified_session')
    if not Path(session.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError('execute each runtime in a separate process')
    classes = {}
    for name, class_name, environment in (
            ('ambiguous_instrument_lab', 'AmbiguousInstrumentLab', 'ambiguous_instruments'),
            ('dynamic_instrument_lab', 'DynamicInstrumentLab', 'dynamic_instruments')):
        canonical = 'hegel_machine.cognition.' + name
        spec = importlib.util.spec_from_file_location(canonical, ROOT / protocol['shared_laboratories'][name])
        laboratory = importlib.util.module_from_spec(spec)
        sys.modules[canonical] = laboratory
        spec.loader.exec_module(laboratory)
        classes[environment] = getattr(laboratory, class_name)
    def factory(item):
        arguments = dict(item)
        return classes[arguments.pop('environment')](**arguments)
    return session, factory


common.load = load


def summarize(protocol, output, cohort):
    common.summarize(protocol, output, cohort)
    summary = json.loads((output / cohort / 'summary.json').read_text())
    applicability = {}
    for arm, data in summary['arms'].items():
        tasks = []
        for task in data['tasks']:
            state = json.loads(common.run_path(output, cohort, arm, task['apparatus_seed']).read_text())
            events = state['results'][task['position']]['research']['actions']
            tasks.append({key: task[key] for key in ('apparatus_seed', 'position', 'stage', 'solved', 'costs')} | {
                'preparations': [event for event in events if event['action'] == 'prepare_control_revision'],
                'choices': [event for event in events if event['action'] == 'choose_observation_research'],
                'confirmations': [{key: value for key, value in event.items()
                                   if key not in ('program', 'baseline_program')}
                                  for event in events if event['action'] == 'confirm_observation_research']})
        applicability[arm] = {
            'contexts': dict(Counter(choice['context'] for task in tasks for choice in task['choices'])),
            'experienced_choices': sum(choice.get('trials', 0) > 0 for task in tasks for choice in task['choices']),
            'tasks': tasks}
    common.save(output / cohort / 'applicability_digest.json', applicability)


def queue(protocol, output, cohort):
    jobs = [('learn', arm, seed) for i, seed in enumerate(protocol['cohorts'][cohort])
            for arm in (('learned', 'policy_disabled', 'baseline_v093') if cohort == 'development'
                        else ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)])]
    jobs.extend(('evaluate', arm, seed) for seed in protocol['cohorts'][cohort] for arm in ARMS)
    execution = {'status': 'running', 'completed': []}
    for mode, arm, seed in jobs:
        log = output / cohort / 'logs' / f'{mode}_{arm}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        common.save(output / cohort / 'execution.json', {**execution, 'active': [mode, arm, seed]})
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--output', str(output), '--cohort', cohort, '--arm', arm, '--seed', str(seed)],
                stdout=stream, stderr=subprocess.STDOUT)
        if code:
            common.save(output / cohort / 'execution.json', {**execution, 'status': 'failed', 'exit_code': code})
            raise SystemExit(code)
        execution['completed'].append([mode, arm, seed])
        print(json.dumps({'finished': [mode, arm, seed]}), flush=True)
    summarize(protocol, output, cohort)
    common.save(output / cohort / 'execution.json', {**execution, 'status': 'completed', 'exit_code': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'learn', 'evaluate', 'queue', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cohort', choices=('development', 'main'), default='development')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if args.mode == 'freeze':
        common.freeze(args.output)
    elif args.mode == 'queue':
        queue(protocol, args.output, args.cohort)
    elif args.mode == 'summarize':
        summarize(protocol, args.output, args.cohort)
    elif args.mode == 'learn':
        common.learn(protocol, args.output, args.cohort, args.arm, args.seed)
    else:
        common.evaluate(protocol, args.output, args.cohort, args.arm, args.seed)


if __name__ == '__main__':
    main()
