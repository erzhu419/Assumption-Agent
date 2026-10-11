"""Check retained source cases using the existing zero-model world learner."""
import json
from pathlib import Path

from hegel_machine.cognition.world_method_learning import WorldMethodLearner
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_discoveryworld_pilot import SELECTION, write_json

output = Path(__file__).resolve().parent
bundle = json.loads(Path('artifacts/discoveryworld_estimator_interface_20261008/candidate_bundle.json').read_text())
bundle['files'][METHOD_PATH] = (output / 'candidate.py').read_text()
source = Path(bundle['source'])
plan = json.loads((source.parents[1] / 'protocol.json').read_text())
source_result = json.loads((source / 'result.json').read_text())
task = next(t for t in plan['tasks'] if t['id'] == source_result['task_id'])
records = json.loads((source / 'agent/laboratory_observations.json').read_text())
archive = json.loads(bundle['files']['shared/source_episode.json'])
registered = {t['id']: t for t in json.loads(SELECTION.read_text())['tasks']}
agent = WorldMethodLearner(None, output / 'old_cases', knowledge={'files': bundle['files']},
    task=task, records=records, prefix_steps=source_result['prefix_environment_steps'],
    public_spec=archive['public_spec'], limits=plan['limits'], calls=48,
    runtime=plan['runtime'], provider_factory=lambda _: None, max_steps=256,
    additional_tasks=[registered[i] for i in ('dw_04_normal_s2', 'dw_04_normal_s4')])
rows = []
for case in agent.cases:
    trial = agent._practice(case)
    raw = json.loads((agent.workspace / trial['directory'] / 'result.json').read_text())
    rows.append({'case': case, 'trial': trial['directory'], 'status': trial['status'],
        'completedSuccessfully': raw['score']['completedSuccessfully'],
        'new_environment_steps': raw['new_environment_steps'],
        'replayed_environment_steps': raw['replayed_environment_steps'],
        'model_calls': raw['costs']['model_calls']})
    write_json(output / 'old_case_results.json', rows)
    print(json.dumps(rows[-1]), flush=True)
