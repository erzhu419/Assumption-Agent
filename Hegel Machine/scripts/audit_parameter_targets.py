"""Audit retained parameter targets without generating, predicting or fitting.

Only the original sparse-joint views and D1 results are inspected. Source
Programs are small retained inputs; paid trajectories and old check tables stay
at their original references. The caller freezes the runtime before execution.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import monotonic

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.parameter_roles import (
    align_parameter_roles, describe_bound_parameter_roles,
)
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.revision_parameter_relations import (
    diagnose_parameter_targets, target_conflicts,
)
from hegel_machine.cognition.revision_relations import _match_parent


SOURCE_SCOPES = (6, 7, 8)


def reference(path, *parts):
    return {'file': str(path), 'json_path': list(parts)}


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def spend(budget, amount=1):
    if not budget.spend(amount):
        raise ValueError('parameter_target_audit_budget_exhausted')


def retained_views(search, budget):
    """Index exactly the saved views, retaining every original attempt address."""
    views = {}
    for index, attempt in enumerate(search['attempts']):
        spend(budget)
        names = tuple(sorted([*attempt['input_holes'], *attempt['evidence_inputs']]))
        key = attempt['output_hole'], names
        if key not in views:
            views[key] = {'output_hole': key[0], 'input_names': list(names), 'attempt_indices': []}
        views[key]['attempt_indices'].append(index)
    return list(views.values())


def view_conflicts(scope, views, tolerance, budget):
    """Training witnesses can prune; held-inclusive comparisons only diagnose."""
    features = [record['features'] for record in scope['records']]
    results = []
    for view in views:
        output = view['output_hole']
        targets = [record['targets'][output] for record in scope['records']]
        training = target_conflicts(features, targets, scope['training_indices'],
                                    view['input_names'], tolerance, budget)
        held = target_conflicts(features, targets, range(len(features)),
                                view['input_names'], tolerance, budget)
        results.append({**view, 'training': training, 'held_inclusive_diagnostic': held,
            'prune_from_training_conflict': training['status'] == 'conflict',
            'held_used_for_admission': False,
            'sample_gate_order': 'Training conflict diagnosis precedes the sample/free-parameter guard.'})
    return results


def fixed_inputs(searches, candidates, metadata, fit_states, workspace, budget):
    """Resolve the existing endpoints once and reuse exact joint substitutions."""
    revisions = fit_states['revisions']
    refs = []
    for revision in revisions:
        for key in ('before_ref', 'after_ref'):
            if revision[key] not in refs:
                refs.append(revision[key])
    programs = [Program.from_dict(item['value']).expanded()
                for item in load_source_values(refs, workspace)]

    def endpoint(ref):
        return programs[refs.index(ref)]

    records_by_scope, source_records = {}, {}
    for index, search in searches.items():
        candidate, scope = candidates[search['candidate_id']], metadata[str(index)]
        if len(scope['records']) != len(search['source_records']):
            raise ValueError('saved_metadata_source_record_count_changed')
        before_pattern = Program.from_dict(candidate['before_pattern'])
        after_pattern = Definition.from_dict(candidate['definition']).body
        records, compact = [], []
        for row_index, (old, numeric) in enumerate(zip(search['source_records'], scope['records'])):
            matches = [(i, revision) for i, revision in enumerate(revisions)
                       if revision['before_ref'] == old['before_ref']
                       and revision['after_ref'] == old['after_ref']]
            if len(matches) != 1 or any(numeric[key] != old[key]
                                       for key in ('source_id', 'before_ref', 'after_ref')):
                raise ValueError('unique_original_fit_state_revision_required')
            revision_index, revision = matches[0]
            before, after = endpoint(old['before_ref']), endpoint(old['after_ref'])
            bindings, before_paths = _match_parent(before_pattern, before, candidate['holes'], budget)
            bindings, after_paths = _match_parent(after_pattern, after, candidate['holes'], budget,
                                                 bindings=bindings)
            for name, expected in {**numeric['features'], **numeric['targets']}.items():
                if name.startswith('evidence:'):
                    continue
                spend(budget)
                if bindings[name].op != 'const' or float(bindings[name].value) != expected:
                    raise ValueError('retained_metadata_parameter_binding_changed')
            records.append({**deepcopy(old), 'fit_state_refs': deepcopy(revision['fit_state_refs']),
                'bindings': bindings, '_before_program': before, '_after_program': after,
                '_before_binding_paths': before_paths, '_after_binding_paths': after_paths,
                '_roles': deepcopy(candidate['fixed_role_signature']),
                '_output_unit': candidate['output_unit']})
            compact.append({'source_record_index': row_index,
                'source_record_ref': reference(search['_summary_file'], 'feedback',
                    'parameter_relation_searches', index, 'source_records', row_index),
                'before_program_index': refs.index(old['before_ref']),
                'after_program_index': refs.index(old['after_ref']),
                'fit_state_revision_index': revision_index})
        records_by_scope[index], source_records[str(index)] = records, compact
    return records_by_scope, {'program_endpoints': [
        {'source_ref': ref, 'program': program.to_dict()} for ref, program in zip(refs, programs)],
        'source_records': source_records}


def full_input_views(scope):
    names = sorted(scope['records'][0]['features'])
    return [{'output_hole': name, 'input_names': names, 'attempt_indices': []}
            for name in scope['records'][0]['targets']]


def resolve_d1_cases(all_audit, pilot):
    """D1's reused pilot is evidence, not another fit or another debit."""
    cases = []
    for entry in all_audit['cases']:
        if 'reused_from' not in entry:
            cases.append(entry)
            continue
        original, = [case for case in pilot['cases'] if case['case_id'] == entry['case_id']]
        cases.append(original)
    return cases


def d1_strata(cases, searches, output_contracts, input_contracts, budget):
    result = []
    for case in cases:
        spend(budget)
        scope_index, attempt_index = map(int, case['case_id'].split(':'))
        attempt = searches[scope_index]['attempts'][attempt_index]
        output = attempt['output_hole']
        contract = output_contracts[scope_index][output]
        bad_inputs = [name for name in attempt['input_holes'] if name not in input_contracts[scope_index]]
        result.append({'case_id': case['case_id'], 'output_hole': output,
            'target_status': contract['status'], 'target_reason': contract.get('reason'),
            'input_role_status': 'UNKNOWN' if bad_inputs else 'comparable',
            'unmatched_parent_input_holes': bad_inputs,
            'full_view_role_status': ('comparable' if contract['status'] == 'comparable'
                                      and not bad_inputs else 'UNKNOWN'),
            'D1_observed_outcome': case['observed_outcome'],
            'D1_parameter_identifiability': case['parameter_identifiability'],
            'D1_training_rejection_removed': case['ridge_training_rejection_removed'],
            'scope': 'Role comparability, numeric training fit and development held checks are separate.'})
    return result


def revision_alignments(candidates, records_by_scope, budget):
    """Inspect each actual replacement once, including unknown/ambiguous roles."""
    seen, results = [], []
    for index, records in records_by_scope.items():
        for record in records:
            identity = record['before_ref'], record['after_ref']
            if identity in seen:
                continue
            seen.append(identity)
            roles = describe_bound_parameter_roles(candidates[index], record, budget)['roles']
            before = {name: item for name, item in roles.items() if item.get('side') == 'before'}
            after = {name: item for name, item in roles.items() if item.get('side') == 'after'}
            alignment = align_parameter_roles(before, after, budget)
            for mapping in alignment['mappings']:
                mapping['before_parameter_paths'] = before[mapping['source_path']]['parameter_paths']
                mapping['after_parameter_paths'] = after[mapping['target_path']]['parameter_paths']
            results.append({'before_ref': record['before_ref'], 'after_ref': record['after_ref'],
                'source_id': record['source_id'], 'alignment': alignment,
                'scope': 'Unique syntax-derived basis/scale correspondence, not an edit law or new support.'})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--metadata-rows', required=True, type=Path)
    parser.add_argument('--source-fit-state', required=True, type=Path)
    parser.add_argument('--d1-all', required=True, type=Path)
    parser.add_argument('--d1-pilot', required=True, type=Path)
    parser.add_argument('--runtime-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--compute-budget', type=int, default=250000)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('Use a new output directory; preserve earlier evidence.')
    started = monotonic()
    source_path = args.summary.resolve()
    source = load(source_path)  # The single permitted local parse of the old summary.
    searches = {i: source['feedback']['parameter_relation_searches'][i] for i in SOURCE_SCOPES}
    if any(search['input_scope'] != 'sparse_parent_evidence_pairs' for search in searches.values()):
        raise ValueError('original_sparse_joint_scopes_required')
    for search in searches.values():
        search['_summary_file'] = str(source_path)
    library_ref = source['library_ref']
    library, = load_source_values([library_ref], source_path.parent)
    candidates = {search['candidate_id']: library['value']['candidates'][search['candidate_id']]
                  for search in searches.values()}
    metadata, fit_states = load(args.metadata_rows), load(args.source_fit_state)
    budget = SearchBudget(args.compute_budget)
    ledger = {}
    before = budget.spent
    records_by_scope, source_pack = fixed_inputs(searches, candidates, metadata, fit_states,
                                                source_path.parent, budget)
    ledger['exact_source_binding'] = budget.spent - before
    output_contracts, input_contracts, reports = {}, {}, []
    for index, search in searches.items():
        before = budget.spent
        diagnosis = diagnose_parameter_targets(candidates[search['candidate_id']],
                                                records_by_scope[index], budget)
        ledger[f'{index}:target_diagnosis'] = budget.spent - before
        eligible = {hole['name'] for hole in diagnosis['output_holes']}
        input_contracts[index] = {hole['name'] for hole in diagnosis['input_holes']}
        excluded = {item['hole']: item for item in diagnosis['excluded']}
        output_contracts[index] = {name: {
            'status': 'comparable' if name in eligible else 'UNKNOWN',
            'reason': (None if name in eligible else excluded.get(name, {}).get(
                'reason', 'common_computational_parameter_role_required'))}
            for name in metadata[str(index)]['records'][0]['targets']}
        before = budget.spent
        views = retained_views(search, budget)
        conflicts = view_conflicts(metadata[str(index)], views, search['contract']['tolerance'], budget)
        for item in conflicts:
            input_holes = [name for name in item['input_names'] if not name.startswith('evidence:')]
            item['unmatched_parent_input_holes'] = [name for name in input_holes
                                                  if name not in input_contracts[index]]
            item['input_role_status'] = ('UNKNOWN' if item['unmatched_parent_input_holes'] else 'comparable')
            item['output_role_status'] = output_contracts[index][item['output_hole']]['status']
            item['full_view_role_status'] = ('comparable' if item['input_role_status'] ==
                                             item['output_role_status'] == 'comparable' else 'UNKNOWN')
        full = view_conflicts(metadata[str(index)], full_input_views(metadata[str(index)]),
                              search['contract']['tolerance'], budget)
        ledger[f'{index}:saved_raw_view_conflicts'] = budget.spent - before
        reports.append({'search_index': index, 'candidate_id': search['candidate_id'],
            'search_ref': reference(source_path, 'feedback', 'parameter_relation_searches', index),
            'metadata_scope_ref': reference(args.metadata_rows.resolve(), str(index)),
            'target_diagnosis': diagnosis, 'output_role_comparison': output_contracts[index],
            'retained_raw_views': conflicts, 'full_original_input_views': full,
            'view_coordinate_scope': 'Original D1 raw constants and evidence coordinates; role-invalid outputs remain excluded by the learner diagnosis.',
            'saved_attempt_count': len(search['attempts']), 'distinct_view_count': len(views),
            'training_conflict_views': sum(item['prune_from_training_conflict'] for item in conflicts),
            'training_conflict_attempts': sum(len(item['attempt_indices']) for item in conflicts
                                            if item['prune_from_training_conflict']),
            'full_input_sufficiency': 'UNKNOWN'})
    before = budget.spent
    alignments = revision_alignments({index: candidates[search['candidate_id']]
        for index, search in searches.items()}, records_by_scope, budget)
    ledger['actual_revision_role_alignments'] = budget.spent - before
    cases = resolve_d1_cases(load(args.d1_all), load(args.d1_pilot))
    before = budget.spent
    strata = d1_strata(cases, searches, output_contracts, input_contracts, budget)
    ledger['D1_saved_result_classification'] = budget.spent - before
    summary = {'scope_count': len(reports), 'saved_attempt_count': sum(
        len(search['attempts']) for search in searches.values()), 'D1_case_count': len(strata),
        'D1_target_status': dict(Counter(case['target_status'] for case in strata)),
        'D1_input_role_status': dict(Counter(case['input_role_status'] for case in strata)),
        'D1_full_view_role_status': dict(Counter(case['full_view_role_status'] for case in strata)),
        'D1_observed_outcomes': dict(Counter(case['D1_observed_outcome'] for case in strata)),
        'D1_joint_strata': dict(Counter(
            f"output={case['target_status']} / input={case['input_role_status']} / {case['D1_observed_outcome']}"
            for case in strata)),
        'actual_retention_revisions': len(alignments),
        'training_conflict_views': sum(report['training_conflict_views'] for report in reports),
        'training_conflict_attempts': sum(report['training_conflict_attempts'] for report in reports),
        'qualified_new_relation': False}
    args.output_dir.mkdir(parents=True)
    source_pack['candidates'] = {name: {key: candidate[key] for key in (
        'id', 'before_pattern', 'definition', 'holes', 'parent_holes', 'fixed_role_signature',
        'output_unit') if key in candidate} for name, candidate in candidates.items()}
    source_pack.update(metadata_rows_ref=str(args.metadata_rows.resolve()),
                       fit_state_audit_ref=str(args.source_fit_state.resolve()))
    save(args.output_dir / 'source_pack.json', source_pack)
    result = {'kind': 'retained_parameter_target_identity_and_view_audit',
        'runtime_manifest_ref': str(args.runtime_manifest.resolve()),
        'source_summary_ref': str(source_path), 'library_ref': library_ref,
        'source_pack_ref': str((args.output_dir / 'source_pack.json').resolve()),
        'metadata_rows_ref': str(args.metadata_rows.resolve()),
        'source_fit_state_ref': str(args.source_fit_state.resolve()),
        'D1_refs': {'all': str(args.d1_all.resolve()), 'pilot': str(args.d1_pilot.resolve())},
        'scopes': reports, 'revision_role_alignments': alignments,
        'D1_strata': strata, 'summary': summary,
        'new_compute_cost': budget.spent, 'compute_limit': budget.limit,
        'charged_work': ledger, 'wall_seconds': monotonic() - started,
        'IO_cost_convention': 'JSON loading, selecting existing records and writing small slices do not execute candidate-row evaluations; structural matching, roles, view indexing and conflict checks debit the shared SearchBudget.',
        'new_predictions': 0, 'new_fits': 0, 'new_observations': 0, 'provider_calls': 0,
        'factory_enumeration': False, 'scientific_support_added': False,
        'scope': 'Same consumed source history. Training conflicts constrain only their declared views; held rows diagnose, never admit or prune. Role UNKNOWN is not a world-law refutation.'}
    save(args.output_dir / 'audit.json', result)
    print(json.dumps({'audit_ref': str(args.output_dir / 'audit.json'), 'summary': summary,
                      'new_compute_cost': budget.spent}))
    return result


if __name__ == '__main__':
    main()
