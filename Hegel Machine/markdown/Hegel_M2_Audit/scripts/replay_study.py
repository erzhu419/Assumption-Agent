"""Invoke the unchanged fixed-commit M2 study with an explicit audit output path."""
from _common import HEGEL_REPO, AUDIT_OUTPUT_DIR
import json
import runpy

study = runpy.run_path(str(HEGEL_REPO / 'scripts/run_v16_8_m2_open_workspace_wormhole_study.py'))
result = study['run_m2_study'](AUDIT_OUTPUT_DIR / 'replay')
print(json.dumps({'output': str(AUDIT_OUTPUT_DIR / 'replay'), 'result_keys': list(result)}, indent=2))
