"""Recompute production digests and inspect two authentic same-AST Pi instances."""
from pathlib import Path
import hashlib
import json
from run_scale_controls import ROOT, OUT, SNAPS, BUNDLE
from hegel_machine.cognition.wormhole_forest import canonical_digest
from hegel_machine.cognition.m4_compositional_discovery import (
    load_verified_authored_method, predict_with_frozen_instance,
)

# ROOT/OUT inherit HEGEL_AUDIT_REPO and HEGEL_AUDIT_OUT from run_scale_controls.
OUT.mkdir(parents=True, exist_ok=True)
checks = {}
for name in ('true_stage0_forest_snapshot.json', 'unmaintained_post_stage1_forest_snapshot.json',
             'post_stage2_forest_snapshot.json'):
    forest = json.loads((SNAPS / name).read_text())
    mismatches = []
    for node_id, nd in forest['nodes'].items():
        # Exactly the five fields used by PhilosophicalForest.register_node.
        actual = canonical_digest({k: nd[k] for k in (
            'task_view', 'observation_map', 'scale_map', 'claim_contract', 'program_steps')})
        if actual != nd['semantic_digest']:
            mismatches.append({'node_id': node_id, 'recorded': nd['semantic_digest'], 'computed': actual})
    cert_checks = []
    for edge in forest['coarse_graining_edges']:
        if 'closure_certificate_digest' not in edge:
            continue
        actual = hashlib.sha256(json.dumps(edge['empirical_closure_certificate'], sort_keys=True).encode()).hexdigest()[:16]
        cert_checks.append({'coarse_node_id': edge['coarse_node_id'],
                            'recorded': edge['closure_certificate_digest'], 'computed': actual,
                            'matches': actual == edge['closure_certificate_digest']})
    checks[name] = {'node_count': len(forest['nodes']),
                    'semantic_digest_mismatches': mismatches, 'certificate_digest_checks': cert_checks}
(OUT / 'archive_digest_checks.json').write_text(json.dumps(checks, indent=2, sort_keys=True))

base = BUNDLE / 'persisted_authored_workspaces'
a0 = load_verified_authored_method(base / 'stage0_metered_sources/stage0_pi_s_schur/authored_method.json',
                                  require_external_manifest=True, verify_frozen_only=True)
a2 = load_verified_authored_method(base / 'stage2_lifecycle/stage2_pi_s_reverified/authored_method.json',
                                  require_external_manifest=True, verify_frozen_only=True)
points = [{'u_drive': 2.0, 'T_horizon': 2.5}, {'u_drive': 2.5, 'T_horizon': 3.5}]
keys = ('model_sha256', 'ast_skeleton_digest', 'fitted_params', 'confirmed_instance_digest')
instances = {'stage0': {k: a0[k] for k in keys}, 'stage2': {k: a2[k] for k in keys},
             'points': points, 'p0': predict_with_frozen_instance(a0, points),
             'p2': predict_with_frozen_instance(a2, points)}
(OUT / 'real_certificate_instances.json').write_text(json.dumps(instances, indent=2, sort_keys=True))
print(json.dumps({'digest_checks': {k: {'node_count': v['node_count'],
                                       'mismatch_count': len(v['semantic_digest_mismatches'])}
                                  for k, v in checks.items()},
                  'same_source': a0['model_sha256'] == a2['model_sha256'],
                  'same_instance': a0['confirmed_instance_digest'] == a2['confirmed_instance_digest']}, indent=2))
