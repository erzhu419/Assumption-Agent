"""Two runtimes resume the same three complete paid states; terminal scoring waits."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_observed_condition_decoding_study import laboratory, load_runtime
from run_knowledge_lineage_study import read, save
COSTS = ('observation_cost', 'measurement_cost', 'compute_cost')


def compact(value):
    omitted = {'program', 'baseline_program', 'reading_mean_program', 'candidate',
               'hypotheses', 'supported_hypotheses', 'alternatives', 'competing_hypotheses'}
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k not in omitted}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    return value


def quote(offer):
    result = compact(offer)
    candidate = offer.get('candidate', {})
    if candidate.get('instrument_condition_experiment'):
        result['instrument_decision'] = compact(candidate['instrument_condition_experiment'].get('decision'))
    if candidate.get('method_opportunity'):
        result['method_opportunity'] = compact(candidate['method_opportunity'])
    return result


def barrier(*_):
    if not read(BASE / 'learning_completion.json')['complete']:
        raise RuntimeError('All three new continuations must finish before terminal scoring.')


def learn(protocol, arm, root):
    module = load_runtime(protocol, arm)
    from hegel_machine import __version__
    directory = BASE / 'runs' / arm / root
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / protocol['prefixes'][root]
    original = read(source)
    session = module.UnifiedResearchSession(state=original, lab_factory=laboratory)
    before_costs = session._costs()
    initial_actions = {r['task_position']: len(r['research']['actions']) for r in original['results']}
    initial_events = len(original['scheduler_events'])
    record = {'complete': False, 'root': root, 'arm': arm, 'runtime_version': __version__,
        'runtime_source': protocol['arms'][arm]['runtime'], 'prefix_source': str(source.relative_to(ROOT)),
        'prefix_costs': before_costs, 'remaining_before': session._remaining(),
        'terminal_scores': 0, 'external_model_calls': 0, 'quotes_after_feedback': [], 'visits': []}
    original_quotes = module.UnifiedResearchSession._offer_response_questions
    original_visit = module.UnifiedResearchSession._perform_revisit

    def observe_quotes(current):
        start = current._costs()
        result = original_quotes(current)
        record['quotes_after_feedback'].append({
            'after_total_revisits': len(current.state['revisits']),
            'remaining': current._remaining(),
            'screening_costs': {k: current._costs()[k] - start[k] for k in COSTS},
            'offers': [quote(q) for q in current.state['pending_revisits']]})
        return result

    def observe_visit(current, offer, **kwargs):
        position = offer['task_position']
        previous = current.state['results'][position]['research']
        program = deepcopy(previous['program'])
        action_count = len(previous['actions'])
        before = current._costs()
        started = time.process_time()
        result = original_visit(current, offer, **kwargs)
        report = current.state['results'][position]['research']
        record['visits'].append({'position': position, 'executed': bool(result),
            'selected_quote': quote(offer), 'cpu_seconds': time.process_time() - started,
            'whole_incremental_costs': {k: current._costs()[k] - before[k] for k in COSTS},
            'execution_costs': current.state['revisits'][-1]['visit_costs'] if result else None,
            'program_changed': program != report['program'],
            'paid_events': compact(report['actions'][action_count:])})
        save(directory / 'progress.json', {'complete': False,
            'attempted_visits': len(record['visits']), 'costs': current._costs(),
            'remaining': current._remaining()})
        return result

    module.UnifiedResearchSession._offer_response_questions = observe_quotes
    module.UnifiedResearchSession._perform_revisit = observe_visit
    cpu, wall = time.process_time(), time.monotonic()
    state = session.run()
    elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.monotonic() - wall
    record.update(complete=state['status'] == 'completed', remaining_after=session._remaining(),
        suffix_costs={k: session._costs()[k] - before_costs[k] for k in COSTS},
        all_learning_costs=session._costs(), suffix_cpu_seconds=elapsed_cpu,
        wall_seconds=elapsed_wall, scheduler_events=compact(state['scheduler_events'][initial_events:]),
        action_counts=dict(Counter(e['action'] for r in state['results']
            for e in r['research']['actions'][initial_actions[r['task_position']]:])))
    state['study'].update(learning_complete=record['complete'],
        cpu_seconds=state['study']['cpu_seconds'] + elapsed_cpu,
        elapsed_seconds=state['study']['elapsed_seconds'] + elapsed_wall,
        steps=state['study']['steps'] + len(record['visits']))
    payload = gzip.compress(json.dumps(state, separators=(',', ':')).encode(), compresslevel=1)
    (directory / 'checkpoint.json.gz').write_bytes(payload)
    save(directory / 'development.json', record)
    save(directory / 'progress.json', {'complete': record['complete'],
        'attempted_visits': len(record['visits']), 'costs': record['all_learning_costs'],
        'remaining': record['remaining_after']})
    if not record['complete']:
        raise RuntimeError('Natural continuation did not complete.')


def evaluate(protocol, arm, root):
    import run_observed_condition_decoding_study as runner
    import run_research_investment_attribution_study as attribution
    attribution.confirm_learning_complete = barrier
    runner.evaluate(BASE, protocol, arm, root)


def summarize(protocol):
    output = {'complete': True, 'external_model_calls': 0, 'terminal_task_arm_scores': 126, 'new_terminal_scores':42, 'reused_terminal_scores':84,
              'scope': protocol['scope'], 'arms': {}, 'pairs': []}
    indexed = {}
    for arm, settings in protocol['arms'].items():
        total = {'suffix_costs': dict.fromkeys(COSTS, 0), 'all_learning_costs': dict.fromkeys(COSTS, 0),
            'terminal_costs': dict.fromkeys(COSTS, 0), 'suffix_cpu_seconds': 0., 'terminal_cpu_seconds': 0.,
            'executed_visits': 0, 'program_changed_visits': 0, 'fresh_pairs': 0, 'positive_fresh_pairs': 0,
            'negative_fresh_pairs': 0, 'visited_task_positions': {}, 'scheduler_actions': Counter(),
            'selected_local_buy_false': 0, 'selected_predicted_loss_reduction': 0.,
            'portfolio_decisions': 0, 'feedback_queue_refreshes': 0}
        rows = []
        for root in protocol['roots']:
            directory = ROOT / settings['result_path'] / root
            result, scores = read(directory / 'development.json'), read(directory / 'evaluation.json')
            total['suffix_cpu_seconds'] += result['suffix_cpu_seconds']
            for group in ('suffix_costs', 'all_learning_costs'):
                for key in COSTS:
                    total[group][key] += result[group][key]
            total['visited_task_positions'][root] = sorted({v['position'] for v in result['visits'] if v['executed']})
            total['executed_visits'] += sum(v['executed'] for v in result['visits'])
            total['program_changed_visits'] += sum(v['executed'] and v['program_changed'] for v in result['visits'])
            total['scheduler_actions'].update(e['action'] for e in result['scheduler_events'])
            total['portfolio_decisions'] += sum(e['action'] == 'choose_research_portfolio' for e in result['scheduler_events'])
            total['feedback_queue_refreshes'] += sum(q['after_total_revisits'] > 0 for q in result['quotes_after_feedback'])
            for visit in result['visits']:
                quoted = visit['selected_quote']
                total['selected_local_buy_false'] += bool(visit['executed'] and (quoted.get('investment_decision') or {}).get('buy') is False)
                total['selected_predicted_loss_reduction'] += quoted.get('expected_loss_reduction', 0.) if visit['executed'] else 0.
                for event in visit['paid_events']:
                    if event['action'] == 'compare_method_followup_on_fresh_response':
                        total['fresh_pairs'] += 1
                        delta = event['before_loss'] - event['after_loss']
                        total['positive_fresh_pairs'] += delta > 1e-7
                        total['negative_fresh_pairs'] += delta < -1e-7
            for score in scores:
                rows.append({'root': root, 'position': score['position'], 'mse': score['mse']})
                total['terminal_cpu_seconds'] += score['cpu_seconds']
                for key in COSTS:
                    total['terminal_costs'][key] += score['costs'][key]
        total.update(tasks=len(rows), passed=sum(r['mse'] <= protocol['pass_mse'] for r in rows),
            mse=sum(r['mse'] for r in rows) / len(rows),
            last7_passed=sum(r['mse'] <= protocol['pass_mse'] for r in rows if r['position'] >= 7),
            last7_mse=sum(r['mse'] for r in rows if r['position'] >= 7) / 21,
            measured_cpu_seconds=total['suffix_cpu_seconds'] + total['terminal_cpu_seconds'],
            all_costs={key: total['all_learning_costs'][key] + total['terminal_costs'][key] for key in COSTS})
        output['arms'][arm] = total
        indexed[arm] = {(r['root'], r['position']): r for r in rows}
    for key, new in indexed['candidate'].items():
        old = indexed['baseline'][key]
        output['pairs'].append({'root': key[0], 'position': key[1], 'candidate_mse': new['mse'], 'baseline_mse': old['mse']})
    output['admission_pairs'] = [{
        'reference': reference, 'root': key[0], 'position': key[1],
        'admission_mse': new['mse'], 'reference_mse': indexed[reference][key]['mse']}
        for reference in ('baseline', 'candidate')
        for key, new in indexed['portfolio_with_local_admission'].items()]
    save(BASE / 'development_summary.json', output)
    print(json.dumps(output['arms'], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('learn', 'evaluate', 'summarize'), required=True)
    parser.add_argument('--arm', choices=('baseline', 'candidate', 'portfolio_with_local_admission'))
    parser.add_argument('--root', choices=('root01', 'root02', 'root03'))
    args = parser.parse_args()
    protocol = read(BASE / 'development_protocol.json')
    if args.root:
        globals()[args.phase](protocol, args.arm, args.root)
        return
    if args.phase == 'summarize':
        summarize(protocol)
        return
    if args.phase == 'evaluate':
        barrier()
    jobs = [(arm, root) for root in protocol['roots'] for arm, settings in protocol['arms'].items() if settings['result_source'] == 'new run']
    def run(job):
        arm, root = job
        directory = BASE / 'runs' / arm / root
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / (args.phase + '.log')).open('w') as stream:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--phase', args.phase,
                '--arm', arm, '--root', root], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        result = {'arm': arm, 'root': root, 'returncode': process.returncode}
        print(result, flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(run, jobs))
    save(BASE / (args.phase + '_jobs.json'), records)
    if any(r['returncode'] for r in records):
        raise SystemExit(1)
    if args.phase == 'learn':
        save(BASE / 'learning_completion.json', {'complete': True, 'new_continuations': 3,
             'terminal_task_arm_scores': 126, 'new_terminal_scores':42, 'reused_terminal_scores':84})


if __name__ == '__main__':
    main()
