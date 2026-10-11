"""Execute the declared frozen cohort once, with all learning before terminal."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[2]
output = Path(__file__).resolve().parent
protocol = json.loads((output / 'protocol.json').read_text())
runner = root / 'scripts/run_action_confirmation_study.py'
ledger = {'started_at': datetime.now(timezone.utc).isoformat(), 'jobs': [],
          'formal_tasks_dispatched': 0, 'status': 'learning'}


def run(mode, arm=None, seed=None):
    command = [sys.executable, str(runner), mode, '--output', str(output)]
    label = mode
    if arm is not None:
        command += ['--arm', arm, '--seed', str(seed)]
        label += '_' + arm + '_' + str(seed)
    record = {'mode': mode, 'arm': arm, 'seed': seed, 'status': 'running'}
    ledger['jobs'].append(record)
    if mode == 'learn':
        checkpoint = output / 'runs' / arm / str(seed) / 'checkpoint.json'
        if checkpoint.exists() and len(json.loads(checkpoint.read_text())['results']) == len(protocol['stages']):
            record['status'] = 'completed_before_queue'
            (output / 'execution.json').write_text(json.dumps(ledger, indent=2) + '\n')
            print(label, 'already learned; no dispatch', flush=True)
            return
    (output / 'execution.json').write_text(json.dumps(ledger, indent=2) + '\n')
    with (output / (label + '.log')).open('a') as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    record['exit_code'] = result.returncode
    record['status'] = 'completed' if result.returncode == 0 else 'failed'
    if result.returncode:
        ledger['status'] = 'stopped_on_error'
    (output / 'execution.json').write_text(json.dumps(ledger, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print(label, 'complete', flush=True)


run('prepare_sources')
for entry in protocol['learning_order']:
    for arm in entry['arms']:
        run('learn', arm, entry['seed'])
for index, seed in enumerate(protocol['recovery']['source_seeds']):
    arms = list(protocol['arms'])
    for arm in arms[index:] + arms[:index]:
        run('learn_recovery', arm, seed)
ledger['status'] = 'terminal'
for entry in protocol['learning_order']:
    for arm in entry['arms']:
        run('evaluate', arm, entry['seed'])
for seed in protocol['recovery']['source_seeds']:
    for arm in protocol['arms']:
        run('evaluate_recovery', arm, seed)
run('summarize')
ledger.update(status='completed', active_research_processes=[],
              completed_at=datetime.now(timezone.utc).isoformat())
(output / 'execution.json').write_text(json.dumps(ledger, indent=2) + '\n')
protocol.update(status='completed', results='summary.json')
(output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
