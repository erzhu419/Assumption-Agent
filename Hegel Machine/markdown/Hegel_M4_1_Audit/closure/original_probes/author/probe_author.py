"""Narrow, offline counterfactuals against commit 11961f702.

Usage:
  PYTHONPATH='/path/to/Hegel Machine/src' python3 probe_author.py --repo '/path/to/Hegel Machine' --out /tmp/m4-author-probe

All experiment workspaces are outside the repository; no existing source is modified.
The target, pilot, and holdout copy the committed M4 fixture exactly. Reordering
changes only the order of the same six historical operator trees. Metadata
probe uses the identical loader expression at m4_compositional_discovery:1247-1249.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess

from hegel_machine.cognition.m4_compositional_discovery import (
    OpenCompositionalCodeProvider,
    _make_open_lab,
    _max_param_idx,
    apply_structural_combinator,
    compile_operator_spec_to_python,
    evaluate_isolated_python_model,
    make_atomic_operator_spec,
    run_compositional_code_agent_episode,
)
from hegel_machine.cognition.open_workspace_wormhole import canonicalize_python_model_ast


def small_result(r):
    return {
        k: r[k] for k in (
            'status', 'strict_pass', 'confirmed_ok', 'confirm_rmse', 'heldout_rmse',
            'evaluated_candidate_count', 'ran_diagnostic', 'agent_step_count', 'cost'
        )
    } | {
        'family': r['authored_artifact']['operator_spec']['family_hint'],
        'ast_digest': r['ast_info']['ast_skeleton_digest'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    evidence = repo / 'docs/evidence/v17_2_m4_multiscale_open_discovery_20261119/persisted_open_author_artifacts'
    result = {
        'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'two order controls + metadata-only control + altered-AST/intact-Python control + three fixed-vs-refit source scope checks + structural combinator parameter check; not full study',
    }

    # Inspect all 10 persisted workspaces and recompile each stored expression.
    artifact_inventory = []
    for d in sorted(evidence.iterdir()):
        meta = json.loads((d / 'authored_method.json').read_text())
        src = (d / 'authored_model.py').read_text()
        rows = json.loads((d / 'observations.json').read_text())
        prefix = json.loads((d / 'confirmed_training_observations.json').read_text())
        ast = canonicalize_python_model_ast(src)
        lens = {'sign': -1.0, 'scale': 1.25, 'offset': 14.0} if d.name.startswith('target_') else {}
        recompiled = compile_operator_spec_to_python(meta['operator_spec'], role_names=meta['role_names'], lens=lens)
        artifact_inventory.append({
            'workspace': d.name,
            'file_count': len(list(d.iterdir())),
            'training_count': len(prefix),
            'observations_equal_prefix': rows == prefix,
            'persisted_code_equals_recompiled_operator_spec': src == recompiled,
            'sha_matches_metadata': hashlib.sha256(src.encode()).hexdigest() == meta['model_sha256'],
            'digest_matches_metadata': ast['ast_skeleton_digest'] == meta['ast_skeleton_digest'],
            'family': meta['operator_spec']['family_hint'],
            'composition_events': meta['composed_history'],
            'model_imports': [line for line in src.splitlines() if line.startswith('import ') or line.startswith('from ')],
        })
    result['artifact_inventory'] = artifact_inventory

    # Same target, observations and exact six-spec history as committed suite.
    fn = lambda f: 14.0 - 1.25 * (9.2 * f['drive_load'] / (2.3 + f['drive_load'] + 0.13 * f['drive_load']**2) + 0.68 * f['bypass_flow'])
    pilot = [[
        {'bypass_flow': 0.0, 'drive_load': 0.4}, {'bypass_flow': 0.4, 'drive_load': 1.3},
        {'bypass_flow': 0.9, 'drive_load': 2.8}, {'bypass_flow': 0.2, 'drive_load': 5.2},
        {'bypass_flow': 1.2, 'drive_load': 8.0}, {'bypass_flow': 0.6, 'drive_load': 10.8},
    ]]
    confirms = [{'bypass_flow': 0.8, 'drive_load': 3.8}, {'bypass_flow': 0.4, 'drive_load': 9.6}]
    heldout_pts = [
        {'bypass_flow': 0.3, 'drive_load': 0.9}, {'bypass_flow': 1.1, 'drive_load': 2.4},
        {'bypass_flow': 0.5, 'drive_load': 6.0}, {'bypass_flow': 1.3, 'drive_load': 9.0},
        {'bypass_flow': 0.2, 'drive_load': 11.2},
    ]
    heldout = [{'features': p, 'target': fn(p)} for p in heldout_pts]
    names = ['S1_monod', 'W_variational_cap', 'W_pure_shunt', 'W_underdamped_ringing', 'W_fast_mode_excess', 'W_haldane_shunt']
    history = [json.loads((evidence / n / 'authored_method.json').read_text())['operator_spec'] for n in names]
    atomic = [make_atomic_operator_spec('linear_1d'), make_atomic_operator_spec('michaelis_1d', arity=2)]
    def target_episode(tag, specs):
        lab = _make_open_lab(
            task_id=tag, surface_domain='chemical_reactor_inverted_sensor',
            feature_names=('bypass_flow', 'drive_load'),
            feature_bounds={'bypass_flow': (0.0, 2.0), 'drive_load': (0.1, 12.0)},
            tolerance=0.055, cost_per_obs=5.0, latent_fn=fn,
        )
        return run_compositional_code_agent_episode(
            lab=lab, heldout_rows=heldout, tolerance=0.055, workspace_dir=out / tag,
            role_names=['drive_load', 'bypass_flow'], initial_query_batches=pilot,
            atomic_base_specs=atomic, prior_operator_specs=specs, confirm_points=confirms,
            lens_hint={'sign': -1.0, 'scale': 1.25, 'offset': 14.0},
        )
    result['ordinary_order_control'] = {
        'same_multiset': sorted(json.dumps(x, sort_keys=True) for x in history) == sorted(json.dumps(x, sort_keys=True) for x in reversed(history)),
        'committed_order': names,
        'committed_order_result': small_result(target_episode('ordinary_original_order', history)),
        'reversed_order': names[::-1],
        'reversed_order_result': small_result(target_episode('ordinary_reverse_order', history[::-1])),
    }
    meta_only = out / 'metadata_only_artifact.json'
    meta_only.write_text(json.dumps({
        'operator_spec': history[-1], 'confirmed_ok': False, 'model_sha256': 'incorrect',
        'ast_skeleton_digest': 'incorrect', 'skeleton_status': 'UNVERIFIED',
    }))
    # Literal data path used in the committed optional-artifact loader, with no code body or receipt.
    loaded = json.loads(meta_only.read_text())
    result['metadata_only_artifact'] = {
        'source_model_present': False, 'receipt_present': False,
        'confirmed_ok_in_metadata': loaded['confirmed_ok'],
        'metadata_digests_invalid': True,
        'result': small_result(target_episode('metadata_only_artifact_target', [loaded['operator_spec']])),
    }

    retained_python_dir = out / 'altered_ast_with_original_python'
    shutil.copytree(evidence / 'W_haldane_shunt', retained_python_dir, dirs_exist_ok=True)
    altered_path = retained_python_dir / 'authored_method.json'
    altered = json.loads(altered_path.read_text())
    altered['operator_spec'] = make_atomic_operator_spec('linear_1d')
    altered_path.write_text(json.dumps(altered, indent=2))
    altered_loaded = json.loads(altered_path.read_text())
    result['altered_ast_original_python_control'] = {
        'source_model_present': (retained_python_dir / 'authored_model.py').is_file(),
        'source_model_unchanged': (retained_python_dir / 'authored_model.py').read_bytes() == (evidence / 'W_haldane_shunt' / 'authored_model.py').read_bytes(),
        'only_changed_metadata_field': 'operator_spec',
        'original_model_is_haldane_shunt': True,
        'metadata_operator_spec_is_linear': True,
        'result': small_result(target_episode('altered_ast_original_python_target', [altered_loaded['operator_spec']])),
    }

    # Holding the witness-trained predictor fixed versus re-fitting the family on old S1 labels.
    s1_eval = [{'features': {'s': s, 'b': 0.0}, 'target': 6.0 * s / (2.0 + s)} for s in (0.9, 1.8)]
    s1_refit = [{'features': {'s': s, 'b': 0.0}, 'target': 6.0 * s / (2.0 + s)} for s in (0.25, 0.6, 1.0, 1.5, 2.0, 2.4)]
    result['source_scope_check'] = {}
    for name in ('W_haldane_shunt', 'W_variational_cap', 'W_pure_shunt'):
        ws = out / ('scope_' + name)
        shutil.copytree(evidence / name, ws, dirs_exist_ok=True)
        prefix = json.loads((ws / 'confirmed_training_observations.json').read_text())
        fixed = evaluate_isolated_python_model(ws, 'authored_model.py', s1_eval, isolated_prefix_rows=prefix)
        refit = evaluate_isolated_python_model(ws, 'authored_model.py', s1_eval, isolated_prefix_rows=s1_refit)
        result['source_scope_check'][name] = {
            'threshold': 0.045, 'fixed_witness_training_prefix_rmse': fixed['rmse'],
            'fixed_witness_training_prefix_pass': fixed['rmse'] <= 0.045,
            'refit_on_old_s1_labels_rmse': refit['rmse'],
            'refit_on_old_s1_labels_pass': refit['rmse'] <= 0.045,
        }

    # Composing after an oscillatory node: parameter traversal must include cos_mul.param.
    osc = apply_structural_combinator(make_atomic_operator_spec('exp_relax_2d'), 'oscillatory_damping_modulation')
    osc_plus = apply_structural_combinator(osc, 'additive_secondary_shunt')
    ps = []
    def collect_all(n, path='root'):
        if isinstance(n, dict):
            if n.get('op') == 'param':
                ps.append({'path': path, **n})
            for k, v in n.items():
                if isinstance(v, dict): collect_all(v, path + '.' + k)
    collect_all(osc_plus)
    compiled = compile_operator_spec_to_python(osc_plus, role_names=['u', 't'], lens={})
    result['oscillatory_then_shunt_parameter_collision'] = {
        'max_param_idx_returned_for_oscillatory_spec': _max_param_idx(osc),
        'expected_max_param_idx': 2,
        'composed_parameter_nodes': ps,
        'compiled_parameter_declaration': next(x for x in compiled.splitlines() if x.startswith('_PARAMS =')),
        'compiled_core_law': next(x for x in compiled.splitlines() if x.startswith('_CORE_LAW_EXPR =')),
        'ast_status_despite_collision': canonicalize_python_model_ast(compiled)['skeleton_status'],
        'expected_independent_params': 4,
        'actual_distinct_params': len({x['idx'] for x in ps}),
    }

    # Direct stdout intervention verifies the parser is causally active, independent of task label.
    clean = OpenCompositionalCodeProvider(role_names=['s','b'],lens={},initial_query_batches=[],atomic_base_specs=atomic)
    changed = OpenCompositionalCodeProvider(role_names=['s','b'],lens={},initial_query_batches=[],atomic_base_specs=atomic)
    clean._compose_from_diagnostic_stdout('{}')
    changed._compose_from_diagnostic_stdout(json.dumps({'high_drive_turnover_drop': True, 'secondary_active': True}))
    result['stdout_counterfactual'] = {'empty_stdout_new_specs': len(clean.composed_history), 'turnover_and_shunt_new_specs': len(changed.composed_history), 'composed_history': changed.composed_history}

    (out / 'probe_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'artifact_inventory'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
