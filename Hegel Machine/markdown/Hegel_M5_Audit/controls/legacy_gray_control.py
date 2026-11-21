"""Read the unchanged committed M5.0 snapshot, exercise current dispatch at b=1.8."""
from pathlib import Path
from copy import deepcopy
import json
import os
import subprocess

from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import load_location_independent_forest
from hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle import ForestScopeDispatcher
from hegel_machine.cognition.wormhole_forest import authorize_abstraction_use

repo = Path(os.environ.get('HEGEL_AUDIT_REPO_ROOT') or subprocess.check_output(
    ['git', 'rev-parse', '--show-toplevel'], text=True).strip())
prefix = 'Hegel Machine/docs/evidence/v17_7_m5_0_closure_and_continual_sublation_20261120/'
commit = '63b940a03205d0f5b36bf2403d274d74709a17fe'
base = Path(__file__).with_name('legacy_gray_snapshot')
base.mkdir(exist_ok=True)
snap_name = 'relocatable_forest_snapshots/post_m5_online_stream_forest_snapshot.json'
artifact_dir = 'persisted_authored_workspaces/persisted_stage1_authored_workspaces/stage1_W4_monod_shunt/'
paths = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', commit, '--', prefix+artifact_dir], text=True).splitlines()
paths.append(prefix + snap_name)
for path in paths:
    dst = base/path.removeprefix(prefix)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(subprocess.check_output(['git', '-C', str(repo), 'show', commit+':'+path]))

forest = load_location_independent_forest(base/snap_name, base)
nd = forest.nodes['stage1_W4_monod_shunt']
metadata = {"active_arity": 2, "haldane_turnover_shunt_regime": False, "secondary_bleed_active": True}
receipt = ForestScopeDispatcher(forest).dispatch_predict(
    query_id='audit_old_scope_b_180', candidate_tree_id='tree_monod_reactor',
    features={"s": 4.0, "b": 1.8}, target_metadata=metadata)
out = {
    "source_commit": commit,
    "snapshot_node_version": nd['version'],
    "actual_preconditions": nd['claim_contract']['preconditions'],
    "gray_zone_bounds_in_actual_contract": nd['claim_contract'].get('gray_zone_bounds'),
    "actual_dispatch": receipt,
    "scope_only_conclusion": 'b=1.8 lies above old supported bound 1.45; current production dispatcher still accepts it using the old b<=1.85 contract. No target-specific prediction-error claim is made.',
}
Path(__file__).with_name('legacy_gray_control_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(out, ensure_ascii=False, indent=2))
