"""Execute the exact generated shell invocation locally with harmless math module; no Docker/model calls."""
from pathlib import Path
import json, os, subprocess, sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
REPO=Path(os.environ.get('HEGEL_AUDIT_REPO', ROOT/'repo_m691_4001bef56'))
HM=REPO if REPO.name == 'Hegel Machine' else REPO/'Hegel Machine'
sys.path.insert(0,str(HM/'src'))
from hegel_machine.cognition import m6_8_b_prime_capability_bridge as bridge
original_run=subprocess.run
observations=[]
def local_docker_proxy(args, **kwargs):
    assert args[:2]==['docker','exec']
    assert args[-2]=='-lc'
    r=original_run(['bash','-lc',args[-1]],**kwargs)
    observations.append({'exact_command':args[-1],'returncode':r.returncode,'stderr':r.stderr,'stdout':r.stdout})
    return r
with patch.object(bridge.subprocess,'run',local_docker_proxy):
    result=bridge.qualify_staged_wormhole_modules_in_container('local-audit', [{'module_import':'math'}])
out={'exact_shell_replay':observations,'returned_qualification':result}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
