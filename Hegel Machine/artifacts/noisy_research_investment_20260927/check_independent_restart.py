"""Real two-task check: empty boundaries, internal learning, exact paid restart."""
import json
from pathlib import Path
import sys

base = Path(__file__).resolve().parent
root = base.parents[1]
sys.path.insert(0, str(root / 'scripts'))
import run_nonlinear_mechanism_revision as runner

output = base / 'independent_restart_check'
assert not output.exists(), 'This check already has outputs; inspect them instead of rerunning.'
output.mkdir()
protocol = json.loads((base / 'development/protocol.json').read_text())
protocol['arms'] = {'independent': {'runtime': str(base / 'baseline/src'), 'knowledge': 'independent'}}
protocol['roots'] = {'root01': {'source_tasks': [], 'tasks': [
    {'apparatus_seed': 9913602, 'seed': 10313605 + 1009 * i, 'response': 'square',
     'inner_weight': .35 + .3 * i, 'window_width': 1, 'noise_sigma': 0.}
    for i in range(2)]}}
protocol['stages'] = ['check_first', 'check_second']
protocol['pass_mse'] = 1e-7
runner.save(output / 'protocol.json', protocol)
original_save = runner.save

class StopAfterPaidTask(Exception):
    pass

def interrupt_save(path, value):
    original_save(path, value)
    if path.name == 'task00.json.gz':
        raise StopAfterPaidTask()

runner.save = interrupt_save
try:
    runner.learn(output, protocol, 'independent', 'root01')
except StopAfterPaidTask:
    pass
finally:
    runner.save = original_save

directory = output / 'runs/independent/root01'
first_path = directory / 'task00.json.gz'
first_bytes = first_path.read_bytes()
first_mtime = first_path.stat().st_mtime_ns
runner.learn(output, protocol, 'independent', 'root01')
state = runner.read(directory / 'checkpoint.json.gz')
tasks = [runner.read(directory / f'task{i:02d}.json.gz') for i in range(2)]
assert first_path.read_bytes() == first_bytes and first_path.stat().st_mtime_ns == first_mtime
assert all(task['results'][0]['available_concepts'] == [] for task in tasks)
assert all(len(task['knowledge']['concepts']) > 0 for task in tasks)
assert all(task['config']['use_memory'] for task in tasks)
assert state['study']['learning_complete'] and not state['revisits']
assert state['costs'] == {key: sum(task['costs'][key] for task in tasks) for key in runner.COSTS}
checkpoint_bytes = (directory / 'checkpoint.json.gz').read_bytes()
runner.learn(output, protocol, 'independent', 'root01')
assert (directory / 'checkpoint.json.gz').read_bytes() == checkpoint_bytes
runner.evaluate(output, protocol, 'independent', 'root01')
result = {'passed': True, 'separate_empty_incoming_libraries': True,
          'within_task_memory_active': True,
          'learned_concepts': [len(task['knowledge']['concepts']) for task in tasks],
          'interruption_after_first_paid_task_did_not_repeat_it': True,
          'completed_restart_did_not_repeat': True, 'costs': state['costs'],
          'terminal_mse': [row['mse'] for row in runner.read(directory / 'evaluation.json')]}
runner.save(output / 'result.json', result)
print(result)
