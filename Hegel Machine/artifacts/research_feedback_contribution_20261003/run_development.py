"""Two real paid prefixes: preserve source execution, remove only its gain credit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
os.environ['HEGEL_PORTFOLIO_OUTPUT'] = str(BASE)
spec = importlib.util.spec_from_file_location('portfolio_resume_runner',
    BASE.parent / 'research_portfolio_20261003' / 'run_development.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def learn(protocol, arm, root):
    original_loader = runner.load_runtime
    audit = {'arm': arm, 'root': root, 'source_position': protocol['source_positions'][root],
             'source_events': [], 'source_expected_gain_components': [],
             'target_withdrawal_predicate_calls': {}, 'target_withdrawal_positive_calls': {},
             'target_counterexample_producer_calls': 0, 'intervention_complete': False}

    def load(*args):
        module = original_loader(*args)
        from hegel_machine.cognition.action_policy import ResearchActionPolicy
        from hegel_machine.cognition.condition_marginal_value import CONDITION_FOLLOWUP_ACTION
        from hegel_machine.cognition.memory import KnowledgeLibrary
        update = ResearchActionPolicy.update
        active, contributions = {'enabled': False}, {}

        def observe_update(policy, action_name, **kwargs):
            context = kwargs.get('context') or policy.action_context(action_name)
            previous = policy._store.get(context, {}).get(action_name, {})
            old_gain = previous.get('expected_gain', 0.)
            result = update(policy, action_name, **kwargs)
            if active['enabled'] and action_name == CONDITION_FOLLOWUP_ACTION:
                multiplier = 0. if result['trials'] == 1 else .5
                added = result['expected_gain'] - multiplier * old_gain
                contributions[context] = multiplier * contributions.get(context, 0.) + added
                audit['source_events'].append({'context': context, 'trial': result['trials'],
                    'gain': result['last_gain'], 'added_expected_gain': added,
                    'before_loss': kwargs['before_loss'], 'after_loss': kwargs['after_loss']})
            return result
        ResearchActionPolicy.update = observe_update

        for name in ('transfer_refuted', 'draft_transfer_refuted'):
            method = getattr(KnowledgeLibrary, name)
            def observed(library, *args, _method=method, _name=name, **kwargs):
                audit['target_withdrawal_predicate_calls'][_name] = audit['target_withdrawal_predicate_calls'].get(_name, 0) + 1
                result = _method(library, *args, **kwargs)
                if result:
                    audit['target_withdrawal_positive_calls'][_name] = audit['target_withdrawal_positive_calls'].get(_name, 0) + 1
                return result
            setattr(KnowledgeLibrary, name, observed)
        producer = KnowledgeLibrary.record_transfer_counterexample
        def observed_producer(library, *args, **kwargs):
            audit['target_counterexample_producer_calls'] += 1
            return producer(library, *args, **kwargs)
        KnowledgeLibrary.record_transfer_counterexample = observed_producer
        parent = module.UnifiedResearchSession

        class SourceGainIntervention(parent):
            def _perform_revisit(self, offer, **kwargs):
                active['enabled'] = (not audit['intervention_complete']
                    and offer['task_position'] == audit['source_position'])
                result = super()._perform_revisit(offer, **kwargs)
                active['enabled'] = False
                return result

            def _offer_new_knowledge(self, position, *args, **kwargs):
                if active['enabled'] and position == audit['source_position']:
                    active['enabled'] = False
                    report = self.state['results'][position]['research']
                    audit['source_execution'] = {
                        'program': deepcopy(report['program']),
                        'visit_costs': deepcopy(report['visit_costs']),
                        'paid_rows': sum(len(batch['observations']) for batch in report['batches']),
                        'batch_probe_ids': [batch.get('probe_id') for batch in report['batches']],
                        'validation_loss': report.get('validation_loss')}
                    for context, component in contributions.items():
                        record = self.library.action_experience[context][CONDITION_FOLLOWUP_ACTION]
                        before = record['expected_gain']
                        if arm == 'candidate':
                            record['expected_gain'] -= component
                        audit['source_expected_gain_components'].append({'context': context,
                            'source_component': component, 'actual_before': before,
                            'consumer_after': record['expected_gain'], 'trials_retained': record['trials']})
                    audit['intervention_complete'] = True
                    audit['after_source_revisits'] = len(self.state['revisits'])
                    audit['intervention_boundary'] = 'after source report and visit costs; before first downstream new-knowledge and response-question quotation'
                return super()._offer_new_knowledge(position, *args, **kwargs)
        module.UnifiedResearchSession = SourceGainIntervention
        return module

    runner.load_runtime = load
    runner.learn(protocol, arm, root)
    runner.save(BASE / 'runs' / arm / root / 'feedback_intervention.json', audit)
    if not audit['intervention_complete']:
        raise RuntimeError('The prespecified source visit was not reached; do not interpret as a tested gain intervention.')


def main():
    protocol = runner.read(BASE / 'development_protocol.json')
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('learn', 'evaluate', 'summarize'), required=True)
    parser.add_argument('--arm', choices=('baseline', 'candidate'))
    parser.add_argument('--root', choices=tuple(protocol['roots']))
    args = parser.parse_args()
    if args.root:
        (learn if args.phase == 'learn' else runner.evaluate)(protocol, args.arm, args.root)
        return
    if args.phase == 'summarize':
        runner.summarize(protocol)
        return
    if args.phase == 'evaluate':
        runner.barrier()
    jobs = [(arm, root) for root in protocol['roots'] for arm in protocol['arms']]
    def execute(job):
        arm, root = job
        directory = BASE / 'runs' / arm / root
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / (args.phase + '.log')).open('w') as stream:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--phase', args.phase, '--arm', arm, '--root', root], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT)
        receipt = {'root': root, 'arm': arm, 'returncode': process.returncode}
        print(receipt, flush=True)
        return receipt
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(execute, jobs))
    runner.save(BASE / (args.phase + '_jobs.json'), receipts)
    if any(item['returncode'] for item in receipts):
        raise SystemExit(1)
    if args.phase == 'learn':
        runner.save(BASE / 'learning_completion.json', {'complete': True,
            'new_continuations': 4, 'terminal_task_arm_scores': 56})

if __name__ == '__main__':
    main()
