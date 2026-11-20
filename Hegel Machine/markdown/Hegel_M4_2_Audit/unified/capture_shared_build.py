"""Read-only audit: capture the true Stage-0 boundary and original builder result."""
import copy
import json
import pickle
import sys
from pathlib import Path

ROOT = Path('/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified')
SOURCE = Path('/workspace/scratch/3f14ad2da0ed/repo_m42_fb4b23aaf/Hegel Machine/src')
sys.path.insert(0, str(SOURCE))
from hegel_machine.cognition import m4_2_unified_closure_and_pi_lifecycle as m

captured = False

def capture(frame, event, arg):
    global captured
    if not captured and event == 'call' and frame.f_code.co_name == '_run_discovery_ep':
        parent = frame.f_back.f_locals
        stage0 = {
            'forest': copy.deepcopy(parent['forest']),
            'stage0_atomic_specs': copy.deepcopy(parent['s0_specs']),
            'stage0_logs': copy.deepcopy(parent['stage0_logs']),
            'stage0_pi_certificate': copy.deepcopy(parent['pi_s_cert_init']),
        }
        with (ROOT / 'true_stage0_snapshot.pkl').open('wb') as f:
            pickle.dump(stage0, f)
        (ROOT / 'true_stage0_forest.json').write_text(json.dumps(stage0['forest'].__dict__, indent=2, default=str))
        captured = True
        print('Captured true Stage0:', len(stage0['forest'].nodes), 'nodes', flush=True)

sys.setprofile(capture)
try:
    built = m.build_and_evolve_m4_2_cross_pillar_forest(ROOT / 'shared_build')
finally:
    sys.setprofile(None)
with (ROOT / 'shared_built.pkl').open('wb') as f:
    pickle.dump(built, f)
(ROOT / 'post_stage2_forest.json').write_text(json.dumps(built['forest'].__dict__, indent=2, default=str))
(ROOT / 'pre_stage2_forest.json').write_text(json.dumps(built['forest_before_stage2_maintenance'].__dict__, indent=2, default=str))
(ROOT / 'shared_build_metadata.json').write_text(json.dumps({k: v for k, v in built.items() if k not in ('forest', 'forest_before_stage2_maintenance')}, indent=2, default=str))
print('Saved shared build:', ROOT / 'shared_built.pkl', flush=True)
