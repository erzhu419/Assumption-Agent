"""Freeze the candidate exactly once after root approval and its source tests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--version', required=True)
args = parser.parse_args()
version = re.search(r'__version__\s*=\s*["\']([^"\']+)', (ROOT/'src/hegel_machine/__init__.py').read_text()).group(1)
if version != args.version:
    raise ValueError('Candidate version differs from approved version')
shutil.copytree(ROOT/'src', BASE/'runtime/src', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(BASE/'freeze.json').write_text(json.dumps({'version':version,'source':'src',
    'frozen_utc':datetime.now(timezone.utc).isoformat(), 'baseline':'baseline_runtime/src',
    'candidate':'runtime/src','learning_continuations':6,'terminal_task_arm_scores':84,
    'maximum_workers':3,'authorization':'root GO after source tests','external_model_calls':0},indent=2)+'\n')
p=BASE/'development_protocol.json';d=json.loads(p.read_text());d['candidate_version']=version;d['status']='Frozen after root GO; all6 continuations precede84 terminal task scores.';p.write_text(json.dumps(d,indent=2)+'\n')
