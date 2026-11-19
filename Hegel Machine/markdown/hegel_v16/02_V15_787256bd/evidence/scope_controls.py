"""Bounded V15 controls using actual AST function bodies and actual data classes.

This does not run the full research agent or physical simulator. Memory containers
are initialized with explicit small inventories; no data is edited in the bundle.
The archive callback is the unmodified nested production function enclosed in a
minimal closure. External synthesis and acquisition are not invoked.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any


def ast_load(path, names, ns):
    tr = ast.parse(path.read_text())
    selected = [n for n in tr.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    found = {n.name for n in selected}
    if found != set(names):
        raise ValueError((str(path), set(names) - found))
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--output', type=Path, default=Path('scope_controls.json'))
    args = ap.parse_args()
    src = args.bundle.resolve() / 'bundled_sources/src/hegel_machine/cognition'
    pkg_name = '_v15_scope_control_package'
    pkg = ModuleType(pkg_name)
    pkg.__path__ = [str(src)]
    sys.modules[pkg_name] = pkg
    loaded = {}
    for name in ('language', 'models'):
        spec = importlib.util.spec_from_file_location(pkg_name + '.' + name, src / (name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        loaded[name] = mod
    P = loaded['language'].Program
    D = loaded['language'].Definition
    O = loaded['models'].Observation
    Budget = loaded['models'].SearchBudget
    index = ModuleType(pkg_name + '.premise_commitment_index')
    index.__dict__.update({'hashlib': hashlib, 'json': json})
    sys.modules[index.__name__] = index
    ast_load(src / 'premise_commitment_index.py', {'_prior_state_fingerprint', 'compute_semantic_version', 'compute_evidence_archive_fingerprint'}, index.__dict__)
    memory = ModuleType(pkg_name + '.memory')
    memory.__package__ = pkg_name
    memory.__dict__.update({'dataclass': dataclass, 'field': field, 'replace': replace, 'deepcopy': deepcopy, 'Program': P, 'Definition': D, 'Observation': O, 'json': json, 'Any': Any})
    sys.modules[memory.__name__] = memory
    ast_load(src / 'abstractions.py', {'AbstractTemplate'}, memory.__dict__)
    ast_load(src / 'memory.py', {'Concept', 'KnowledgeLibrary', '_called_definitions', 'export_library_object_inventory', 'remove_library_object_version'}, memory.__dict__)
    C = memory.Concept
    T = memory.AbstractTemplate

    def make_library(revoked_sibling=False):
        lib = memory.KnowledgeLibrary.__new__(memory.KnowledgeLibrary)
        def prog(v):
            return P('add', children=(P('feature', name='x'), P('const', value=v)))
        lib.concepts = [
            C('focal', prog(1), .5, 'source_a', 'learned'),
            C('other', prog(2), .5, 'source_b', 'learned', status='revoked' if revoked_sibling else 'available'),
            C('exclusive', prog(3), .5, 'source_c', 'derived', parents=['focal']),
            C('shared', prog(4), .5, 'source_d', 'derived', parents=['focal', 'other']),
        ]
        lib.abstracts = [T(c.concept_id + ':abstract:0', c.concept_id, c.program, ('x',), ('x',), ((1,),), parents=[c.concept_id]) for c in lib.concepts]
        lib.meta_priors = [
            {'source_concept_id': 'focal', 'status': 'supported', 'parameter': 1},
            {'source_concept_id': 'other', 'status': 'revoked' if revoked_sibling else 'supported', 'parameter': 2},
        ]
        lib.definitions = {}
        lib.definition_bindings = {}
        lib.deleted_ids = []
        lib.research_drafts = []
        lib.condition_coverages = []
        lib._active_reconciled_branches = []
        lib._generalized_branch_commitments = []
        lib._raw_reconciliation_evidence_archive = [{'kind': 'source_episode_batch', 'source_concept_id': 'focal', 'observations_serialized': [{'features': {'x': 0}, 'target': 1}], 'commitment': {'solved': True}, 'coefficients': [1], 'radius_value': .9}]
        lib._refresh_definitions()
        return lib

    removal = {}
    for revoked in (False, True):
        lib = make_library(revoked)
        ver = index.compute_semantic_version(lib.meta_priors[0])
        out = memory.remove_library_object_version(lib, 'focal', ver)
        inv = memory.export_library_object_inventory(out)
        shared = next(c for c in out.concepts if c.concept_id == 'shared')
        removal['revoked_sibling' if revoked else 'available_sibling'] = {
            'focal_present': any(c.concept_id == 'focal' for c in out.concepts),
            'exclusive_present': any(c.concept_id == 'exclusive' for c in out.concepts),
            'shared_present': True,
            'shared_parents': shared.parents,
            'other_status': next(c.status for c in out.concepts if c.concept_id == 'other'),
            'shared_program_calls_available': out._calls_available(shared.program, None),
            'shared_inventory': inv['objects']['shared'],
            'raw_rows_preserved': len(out._raw_reconciliation_evidence_archive[0]['observations_serialized']),
            'presolved_commitment_removed': 'commitment' not in out._raw_reconciliation_evidence_archive[0],
            'raw_reconstruction_allowed': out._allow_on_demand_raw_history_reconstruction,
        }
    lib = make_library()
    prior_v1 = deepcopy(lib.meta_priors[0])
    prior_v2 = {**deepcopy(prior_v1), 'parameter': 8}
    lib.meta_priors.append(prior_v2)
    ver1 = index.compute_semantic_version(prior_v1)
    ver2 = index.compute_semantic_version(prior_v2)
    lib._active_reconciled_branches = [{'claim_id': 'branch:focal:v1', 'source_concept_id': 'focal', 'semantic_version': ver1}, {'claim_id': 'branch:focal:v2', 'source_concept_id': 'focal', 'semantic_version': ver2}]
    inv_pre = memory.export_library_object_inventory(lib)
    pruned = memory.remove_library_object_version(lib, 'focal', ver1)
    removal['two_versions_same_cid'] = {
        'requested_version': ver1,
        'other_version': ver2,
        'inventory_version_before': inv_pre['objects']['focal']['semantic_version'],
        'remaining_focal_prior_versions': [index.compute_semantic_version(p) for p in pruned.meta_priors if p.get('source_concept_id') == 'focal'],
        'remaining_focal_branches': [b['claim_id'] for b in pruned._active_reconciled_branches if b.get('source_concept_id') == 'focal'],
        'limitation': 'Counterfactual multi-version inventory; no claim that frozen E1/E2 libraries contain two simultaneous versions of this concept id.',
    }
    lib = make_library()
    unchanged = memory.remove_library_object_version(lib, 'focal', 'incorrect_version')
    removal['wrong_version_noop'] = any(c.concept_id == 'focal' for c in unchanged.concepts)

    b_path = src / 'branch_reconciliation.py'
    branch_ns = {'Observation': O, 'Any': Any, 'deepcopy': deepcopy}
    ast_load(b_path, {'_deserialize_obs_rows'}, branch_ns)
    tree = ast.parse(b_path.read_text())
    callback_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_archived_probe_lookup_fn')
    wrapper = ast.parse('def make_callback(seq_probes, budget):\n    replayed_batches_count = 0\n    consumed_ev_ids = set()\n    used_probe_indices = set()\n').body[0]
    wrapper.body.extend([deepcopy(callback_node), ast.Return(value=ast.Name(id='_archived_probe_lookup_fn', ctx=ast.Load()))])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), wrapper], type_ignores=[])), str(b_path), 'exec'), branch_ns)
    req = SimpleNamespace(probe_id='requested', sequence=({'x': 1},), reset=True)
    archive = [{'evidence_id': 'ev:different', 'requested_probe_id': 'another_request', 'actual_probe_id': 'actual_different', 'probe': {'sequence': [{'x': 99}], 'reset': False}, 'observations_serialized': [{'features': {'x': 99}, 'target': 99}], 'transitions': [], 'program_versions_frozen_before_result': ['original_version']}]
    cb = branch_ns['make_callback'](archive, Budget(100))
    response = cb(req, frozen_program_versions=['new_request_version'])
    probe_reuse = {
        'requested_sequence': list(req.sequence), 'requested_reset': req.reset,
        'archived_sequence': archive[0]['probe']['sequence'], 'archived_reset': archive[0]['probe']['reset'],
        'returned_status': response['status'], 'actual_probe_id': response['actual_probe_id'],
        'returned_rows': [{'features': x.features, 'target': x.target} for x in response['rows']],
        'returned_program_versions': response['program_versions_frozen_before_result'],
        'limitation': 'Actual nested callback body in minimal closure. It proves protocol-mismatch fallback acceptance, not that the frozen E1/E2 histories hit the fallback.',
    }
    archive_same_id = deepcopy(archive)
    archive_same_id[0]['requested_probe_id'] = 'requested'
    probe_reuse['same_id_different_sequence_reset_status'] = branch_ns['make_callback'](archive_same_id, Budget(100))(req)['status']

    base = [{'kind': 'source_episode_batch', 'evidence_id': 'ev:a', 'episode_uid': 'ep:a', 'feature_bounds': {'x': [-2, 2]}, 'radius_value': .9, 'commitment': {'direction': 'right'}, 'observations_serialized': [{'features': {'x': 0}, 'target': 1}]}]
    fp = index.compute_evidence_archive_fingerprint(base)
    hashes = {}
    for key, replacement in [('radius_value', .8), ('commitment', {'direction': 'left'}), ('feature_bounds', {'x': [90, 99]})]:
        changed = deepcopy(base)
        changed[0][key] = replacement
        hashes[key + '_edit_invalidates'] = index.compute_evidence_archive_fingerprint(changed) != fp
    partial_prior = {'source_concept_id': 'rebuilt', 'status': 'supported', 'confirmation_loss': 0.0}
    reported_ver = index.compute_semantic_version(partial_prior)
    partial_prior['reconstruction_compute_cost'] = 123
    version_after_cost = {'reported_before_cost_hash': reported_ver, 'stored_after_cost_hash': index.compute_semantic_version(partial_prior), 'mismatch': reported_ver != index.compute_semantic_version(partial_prior), 'scope': 'Reproduces exact hash-before-field-add sequence at branch_reconciliation.py:4447-4449; no complete reconstruction experiment.'}
    result = {'mode': 'bounded_actual_AST_and_data_class_controls', 'full_agent_or_physics_rerun': False, 'source_sha256': {name: hashlib.sha256((src / name).read_bytes()).hexdigest() for name in ['memory.py', 'branch_reconciliation.py', 'premise_commitment_index.py']}, 'object_removal': removal, 'archived_probe_protocol_matching': probe_reuse, 'evidence_hash_dependencies': hashes, 'reconstructed_version_order': version_after_cost}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
