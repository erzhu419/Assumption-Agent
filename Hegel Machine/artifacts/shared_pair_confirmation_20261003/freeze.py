"""Run once after root GO; freeze the tested 0.174 source for offline cohorts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
version = re.search(r'__version__\s*=\s*["\']([^"\']+)',
    (ROOT / 'src/hegel_machine/__init__.py').read_text()).group(1)
if version != '0.174.0':
    raise ValueError('Expected tested 0.174.0 source, found ' + version)
shutil.copytree(ROOT / 'src', BASE / 'runtime/src',
    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for cohort in ('successor_regression',):
    path = BASE / cohort / 'protocol.json'
    protocol = json.loads(path.read_text())
    protocol['candidate_source_status'] = 'Frozen once after production tests and root GO.'
    protocol['execution']['status'] = 'Ready for 36 learning chains; terminal scoring waits for all to finish.'
    path.write_text(json.dumps(protocol, indent=2) + '\n')
(BASE / 'freeze.json').write_text(json.dumps({
    'version': version, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
    'source': 'src', 'runtime': str((BASE / 'runtime/src').relative_to(ROOT)),
    'base': '0.173 within-visit pair investment retained. Add explicit pair confirmation competition in the shared research pool; separately ablate its queue and its new cost experience feedback.',
    'authorization': 'root GO after stable production tests',
    'new_learning_chains': 36, 'new_terminal_task_arm_scores': 504,
    'external_model_calls': 0,
}, indent=2) + '\n')
