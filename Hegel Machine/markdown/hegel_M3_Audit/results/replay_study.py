"""Read-only replay of committed M3 scientific entry point into an audit directory."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

REPO = Path(os.environ.get('HEGEL_REPO', '/workspace/scratch/3f14ad2da0ed/repo_m3_612d8eea5/Hegel Machine'))
OUT = Path(os.environ.get('AUDIT_OUTPUT_DIR', Path(__file__).resolve().parent / 'replay'))
sys.path.insert(0, str(REPO / 'src'))
entry = runpy.run_path(str(REPO / 'scripts/run_v17_0_m3_hipporag_scale_study.py'))
started = time.perf_counter()
print('Starting original run_v17_0_study, fixed 612d8eea5, independent output.', flush=True)
result = entry['run_v17_0_study'](output_dir=OUT)
print(json.dumps({'wall_seconds': time.perf_counter()-started, 'output_dir': str(OUT),
                  'key_paired_comparisons': result.get('key_paired_comparisons'),
                  'out_of_scope_honesty': result.get('out_of_scope_honesty')}, ensure_ascii=False), flush=True)
