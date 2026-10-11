"""Analyze the already-recorded terminal response; no new world or model calls."""
import json
import os
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[2]
source = Path(__file__).resolve().parent
output = source / 'final_response_analysis'
(output / 'shared').mkdir(parents=True, exist_ok=True)
constructor = runpy.run_path(str(root / 'artifacts/discoveryworld_observed_settings_20261008/run.py'))
program = constructor['revised_source']((source / 'agent/shared/consolidated_method.py').read_text())
(output / 'candidate.py').write_text(program)
(output / 'shared/reference_recalibration.json').write_text(
    (source / 'agent/shared/reference_recalibration.json').read_text())
observations = json.loads((source / 'agent/laboratory_observations.json').read_text())
os.chdir(output)
answer = runpy.run_path(str(output / 'candidate.py'))['answer'](
    {'current_observation': observations[-1]['observation']}, observations, {})
result = {'scope': 'Post-hoc analysis of the actual terminal record; original run retained',
          'new_environment_steps': 0, 'new_model_calls': 0, 'answer': answer}
(output / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
