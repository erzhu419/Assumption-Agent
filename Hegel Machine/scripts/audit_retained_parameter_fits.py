"""Audit the saved sparse-joint fits without generating another structure.

The pilot and complete audit must use the same frozen runtime. Complete runs
reuse the three pilot results; historical Programs and source files stay read
only. Only the actual refit/composition/robustification factories are supported.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import monotonic

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

import numpy as np
from hegel_machine.cognition import abstractions, revision_relations, synthesis
from hegel_machine.cognition.abstractions import AbstractTemplate, fit_parameter_slots
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.research_episodes import load_source_values


PILOT = ((8, 472, 'saved_best_training'), (6, 0, 'first_saved_affine'),
         (8, 50, 'low_variation_affine_coordinate'))


class MissingFixedStructure(ValueError):
    pass


def reference(file, *path):
    return {'file': str(file), 'json_path': list(path)}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load(path):
    return json.loads(Path(path).read_text())


def at(program, path):
    for index in path:
        program = program.children[index]
    return program


def recover_parameter_paths(initial, operation, expected_count, input_names):
    """Recognize the frozen unfitted factories, keeping guard thresholds fixed.

    Initial linear intercepts are zero and coefficients are one. Robust offsets
    and branch values are zero. Observed threshold literals belong to the guard,
    even if their numeric value is also zero or one. No factory is called here.
    """
    if operation not in {'refit', 'composition', 'robustification'}:
        raise MissingFixedStructure('unsupported_saved_factory')
    paths, terms, guards = [], [], []

    def term(node):
        if node.op == 'feature' and node.name in input_names:
            return 1
        if node.op == 'mul' and operation == 'composition':
            degree = sum(term(child) for child in node.children)
            if degree <= 2:
                return degree
        raise MissingFixedStructure('fixed_basis_not_from_supported_saved_factory')

    def walk(node, path=()):
        if node.op == 'add':
            for index, child in enumerate(node.children):
                walk(child, (*path, index))
        elif node.op == 'const' and node.value == 0.:
            paths.append(path)
        elif node.op == 'mul' and len(node.children) == 2:
            choices = [i for i, child in enumerate(node.children)
                       if child.op == 'const' and child.value == 1.]
            if len(choices) != 1:
                raise MissingFixedStructure('initial_linear_coefficient_not_identified')
            coefficient = choices[0]
            term(node.children[1 - coefficient])
            terms.append(node.children[1 - coefficient])
            paths.append((*path, coefficient))
        elif node.op == 'if' and operation == 'robustification':
            guard, yes, no = node.children
            if (guard.op not in {'ge', 'lt'} or len(guard.children) != 2
                    or guard.children[0].op != 'feature'
                    or guard.children[0].name not in input_names
                    or guard.children[1].op != 'const'
                    or yes.op != 'const' or no.op != 'const'
                    or yes.value != 0. or no.value != 0.):
                raise MissingFixedStructure('unsupported_fixed_robust_guard')
            guards.append((*path, 0))
            paths.extend(((*path, 1), (*path, 2)))
        else:
            raise MissingFixedStructure('initial_tree_not_supported_unfitted_grammar')

    walk(initial)
    # The actual factories always supply one linear head, then at most one
    # independent robust offset/conditional offset. Empty/seeded heads differ.
    zero_paths = [p for p in paths if at(initial, p).value == 0.]
    if (len(paths) != expected_count or len(set(paths)) != len(paths)
            or set(initial.required_features()) != set(input_names)
            or len(guards) > 1 or not terms
            or (operation != 'robustification' and len(zero_paths) != 1)
            or (operation == 'robustification' and len(zero_paths) != (3 if guards else 2))):
        raise MissingFixedStructure('saved_free_slot_or_input_contract_mismatch')
    return tuple(paths)


def validate_original_fit(initial, ridge, paths, fitting, training_count):
    if (not fitting.get('search_completed') or fitting.get('budget_exhausted')
            or fitting.get('parameter_scope') != 'declared_const_slots_only'
            or fitting.get('actual_fitted_rows') != training_count
            or fitting.get('actual_compute_cost') != (len(paths) + 3) * training_count):
        raise MissingFixedStructure('saved_fit_scope_or_basis_debit_mismatch')
    selected = set(paths)

    def compare(left, right, path=()):
        if path in selected:
            if right.op != 'const' or not isinstance(right.value, (int, float)):
                raise MissingFixedStructure('saved_fitted_slot_is_not_constant')
            return
        if (left.op, left.value, left.name, left.lag, len(left.children)) != (
                right.op, right.value, right.name, right.lag, len(right.children)):
            raise MissingFixedStructure('saved_fit_changes_fixed_tree_or_threshold')
        for index, (a, b) in enumerate(zip(left.children, right.children)):
            compare(a, b, (*path, index))

    compare(initial, ridge)


def snapshot(path):
    path = Path(path).resolve()
    info = path.stat()
    return {'file': str(path), 'bytes': info.st_size, 'modified_ns': info.st_mtime_ns}


def runtime_sources(manifest):
    return {'manifest_ref': str(Path(manifest).resolve()), 'files': [snapshot(path) for path in (
        __file__, abstractions.__file__, synthesis.__file__, revision_relations.__file__)]}


def reconstruct_metadata(searches, candidates, summary_path, budget):
    """Use the existing exact before/after matcher and retained source loader."""
    references = [r[key] for search in searches.values() for r in search['source_records']
                  for key in ('before_ref', 'after_ref')]
    loaded = iter(load_source_values(references, summary_path.parent))
    scopes = {}
    for index, search in searches.items():
        candidate = candidates[search['candidate_id']]
        pattern = Program.from_dict(candidate['before_pattern'])
        body = Definition.from_dict(candidate['definition']).body
        bindings_rows = []
        for record in search['source_records']:
            before, after = [Program.from_dict(next(loaded)['value']).expanded() for _ in range(2)]
            bindings, _ = revision_relations._match_parent(pattern, before, candidate['holes'], budget)
            revision_relations._match_parent(body, after, candidate['holes'], budget, bindings=bindings)
            bindings_rows.append(bindings)
        constants = {hole['name']: [float(row[hole['name']].value) for row in bindings_rows]
                     for hole in candidate['holes']
                     if all(hole['name'] in row and row[hole['name']].op == 'const'
                            for row in bindings_rows)}
        parents = [name for name in constants if name in candidate['parent_holes']]
        outputs = [name for name in constants if name not in candidate['parent_holes']]
        records = []
        for i, record in enumerate(search['source_records']):
            context = record['evidence_context']
            records.append({'source_record_index': i, 'source_id': record['source_id'],
                'before_ref': record['before_ref'], 'after_ref': record['after_ref'],
                'features': {**{name: constants[name][i] for name in parents},
                             **{name: float(value) for name, value in context['features'].items()}},
                'targets': {name: constants[name][i] for name in outputs},
                'evidence_context_ref': reference(summary_path, 'feedback', 'parameter_relation_searches',
                                                   index, 'source_records', i, 'evidence_context')})
        scopes[str(index)] = {'candidate_id': candidate['id'],
            'search_ref': reference(summary_path, 'feedback', 'parameter_relation_searches', index),
            'training_indices': [r['source_record_index'] for r in search['split']['training']],
            'held_indices': [r['source_record_index'] for r in search['split']['held']],
            'records': records}
    return scopes


def old_metrics(check):
    def measure(rows):
        values = [float(row['residual']) for row in rows]
        return {'max_abs': max(map(abs, values)), 'sum_squares': sum(v * v for v in values),
                'rows': len(values)}
    return {key: measure(check[key]) for key in ('training', 'held')}


def spend(budget, units):
    if not budget.spend(units):
        raise MissingFixedStructure('diagnostic_compute_budget_exhausted')


def pilot_matrix(template, rows, budget):
    """Materialize only the selected tiny matrices, charging their evaluations."""
    p = len(template.parameter_paths)
    spend(budget, (p + 1) * len(rows))
    zero = template.instantiate({}, [0.] * p)
    base = np.asarray([zero.predict(row) for row in rows])
    columns = []
    for index in range(p):
        unit = [0.] * p
        unit[index] = 1.
        program = template.instantiate({}, unit)
        columns.append(np.asarray([program.predict(row) for row in rows]) - base)
    targets = np.asarray([row.target for row in rows])
    return {'design': np.asarray(columns).T, 'shifted_targets': targets - base,
            'base_values': base, 'targets': targets}


def audit_case(search_index, attempt_index, search, scope, paths, summary_path, budget,
               matrix_output=None):
    attempt = search['attempts'][attempt_index]
    initial, old_ridge = [Program.from_dict(c['program']) for c in attempt['checks']]
    training = scope['training_indices']
    held = scope['held_indices']
    target = attempt['output_hole']
    rows = [Observation(dict(scope['records'][i]['features']), scope['records'][i]['targets'][target])
            for i in training]
    # Saved expected labels certify the same before/after parameter view; none
    # of the held or after targets enter the input feature dictionary.
    for label, indices in (('training', training), ('held', held)):
        saved = attempt['checks'][1][label]
        if ([r['source_record_index'] for r in saved] != indices
                or any(r['expected_parameter'] != scope['records'][i]['targets'][target]
                       for r, i in zip(saved, indices))):
            raise MissingFixedStructure('reconstructed_target_or_split_differs_from_saved_fit')
    validate_original_fit(initial, old_ridge, paths, attempt['fitting'], len(rows))
    template = AbstractTemplate(f'd1:{search_index}:{attempt_index}', search['candidate_id'],
                                initial, (), (), paths)
    started = budget.spent
    report = {}
    programs = fit_parameter_slots(template, {}, rows, budget, .5, report=report)
    by_method = dict(zip(report.get('candidate_methods', []), programs))
    linear = report.get('linear_diagnostics', {})
    tolerance = search['contract']['tolerance']
    stable = by_method.get('stable_lstsq')
    held_predictions = []
    if stable is not None:
        spend(budget, len(held))
        for i in held:
            record = scope['records'][i]
            expected = record['targets'][target]
            predicted = stable.predict(Observation(record['features'], expected))
            held_predictions.append({'source_record_index': i, 'expected': expected,
                                     'predicted': predicted, 'residual': predicted - expected})
    train_pass = (linear.get('train_residual_max_abs') <= tolerance
                  if linear.get('status') == 'computed' else None)
    held_pass = all(abs(r['residual']) <= tolerance for r in held_predictions) if held_predictions else None
    observed = ('training_and_held_pass' if train_pass and held_pass else
                'training_pass_held_failed' if train_pass and held_pass is False else
                'training_failed' if train_pass is False else 'UNKNOWN')
    old = old_metrics(attempt['checks'][1])
    result = {'case_id': f'{search_index}:{attempt_index}',
        'source_ref': reference(summary_path, 'feedback', 'parameter_relation_searches', search_index,
                                'attempts', attempt_index),
        'initial_program_ref': reference(summary_path, 'feedback', 'parameter_relation_searches',
                                         search_index, 'attempts', attempt_index, 'checks', 0, 'program'),
        'old_ridge_program_ref': reference(summary_path, 'feedback', 'parameter_relation_searches',
                                           search_index, 'attempts', attempt_index, 'checks', 1, 'program'),
        'old_fitting_ref': reference(summary_path, 'feedback', 'parameter_relation_searches', search_index,
                                     'attempts', attempt_index, 'fitting'),
        'operation': attempt['operation'], 'output_hole': target,
        'input_group': attempt['input_group'], 'parameter_paths': [list(p) for p in paths],
        'fixed_thresholds_preserved': True, 'original_ridge': old,
        'initial_reproduced_exactly': by_method.get('initial') == initial,
        'ridge_reproduced_exactly': by_method.get('ridge_lstsq') == old_ridge,
        'fitting_report': {k: v for k, v in report.items() if k != 'losses'},
        'returned_parameters': {method: [at(program, path).value for path in paths]
                                for method, program in by_method.items()},
        'stable_held_predictions': held_predictions,
        'stable_training_pass': train_pass, 'stable_held_pass': held_pass,
        'observed_outcome': observed,
        'parameter_identifiability': report.get('parameter_identifiability', 'UNKNOWN'),
        'ridge_training_rejection_removed': old['training']['max_abs'] > tolerance and train_pass is True,
        'ridge_both_splits_rejection_removed': (old['training']['max_abs'] > tolerance
            and old['held']['max_abs'] > tolerance and train_pass is True and held_pass is True),
        'linear_span_necessary_condition': {
            'least_squares_matrix_sse': linear.get('linear_l2_residual_sum_squares'),
            'all_point_tolerance_sse_upper_bound': len(rows) * tolerance * tolerance,
            'ls_sse_exceeds_necessary_bound': (linear['linear_l2_residual_sum_squares'] >
                len(rows) * tolerance * tolerance if 'linear_l2_residual_sum_squares' in linear else None),
            'nonreachability_certified': False,
            'scope': 'An LS returned instance is evaluated here. A structure-level impossibility claim additionally requires a numerically reliable lower-bound argument; max-absolute residual alone is not an L-infinity optimum.'},
        'new_compute_cost': None, 'scientific_support_added': False,
        'independent_confirmation': False}
    if matrix_output is not None:
        arrays = pilot_matrix(template, rows, budget)
        arrays['old_ridge_parameters'] = np.asarray([at(old_ridge, p).value for p in paths])
        if stable is not None:
            arrays['stable_parameters'] = np.asarray([at(stable, p).value for p in paths])
        np.savez_compressed(matrix_output, **arrays)
        result['matrix_ref'] = str(matrix_output)
        result['matrix_reconstruction_charged_compute'] = (len(paths) + 1) * len(rows)
        result['fixed_programs'] = {'initial': initial.to_dict(),
            'original_ridge': old_ridge.to_dict(),
            'stable': stable.to_dict() if stable is not None else None}
        result['fitting_provenance'] = {
            'metadata_training_weights': 'uniform, original unweighted mean squared error',
            'initial_parameters': list(template.parameters),
            'ridge_penalty': 1e-9,
            'original_world_fit_weights_and_initialization': 'UNKNOWN_not_reconstructed_by_this_D1',
            'original_world_program_refs': [scope['records'][i]['before_ref'] for i in training]}
    result['new_compute_cost'] = budget.spent - started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--runtime-manifest', required=True, type=Path)
    parser.add_argument('--stage', choices=('pilot', 'all'), default='pilot')
    parser.add_argument('--reuse-pilot', type=Path)
    parser.add_argument('--compute-budget', type=int, default=1000000)
    args = parser.parse_args(argv)
    started = monotonic()
    if args.stage == 'all' and not args.reuse_pilot:
        parser.error('The complete audit requires --reuse-pilot; the pilot fits must not repeat.')
    if args.output_dir.exists():
        parser.error('Use a new output directory to preserve completed or interrupted evidence.')
    source_path = args.summary.resolve()
    source = load(source_path)
    all_searches = source['feedback']['parameter_relation_searches']
    selected = {i: all_searches[i] for i in (6, 7, 8)}
    if any(s.get('input_scope') != 'sparse_parent_evidence_pairs' for s in selected.values()):
        parser.error('Only the original three retained sparse-joint scopes are supported.')
    provenance = {'summary': snapshot(source_path), 'library': snapshot(source['library_ref']['file']),
                  'runtime': runtime_sources(args.runtime_manifest)}
    budget = SearchBudget(args.compute_budget)
    preflight, recovered = [], {}
    for si, search in selected.items():
        for ai, attempt in enumerate(search['attempts']):
            if not attempt.get('fitting'):
                continue
            try:
                if len(attempt['checks']) != 2:
                    raise MissingFixedStructure('original_initial_and_ridge_checks_required')
                initial, ridge = [Program.from_dict(c['program']) for c in attempt['checks']]
                paths = recover_parameter_paths(initial, attempt['operation'],
                    attempt['free_parameter_count'], attempt['input_group'])
                validate_original_fit(initial, ridge, paths, attempt['fitting'], len(search['split']['training']))
                recovered[si, ai] = paths
            except (MissingFixedStructure, KeyError) as error:
                preflight.append({'search_index': si, 'attempt_index': ai, 'status': 'MISSING_FIELD',
                    'reason': str(error), 'source_ref': reference(source_path, 'feedback',
                        'parameter_relation_searches', si, 'attempts', ai)})
    prior, reused = None, []
    if args.reuse_pilot:
        prior = load(args.reuse_pilot)
        if prior['provenance'] != provenance or prior['stage'] != 'pilot':
            parser.error('Pilot runtime or exact retained source snapshot differs.')
        metadata_ref = Path(prior['metadata_rows_ref'])
        scopes = load(metadata_ref)
        reused = prior['cases']
        if {case['case_id'] for case in reused} != {f'{si}:{ai}' for si, ai, _ in PILOT}:
            parser.error('The three preregistered pilot case results are required for reuse.')
    else:
        candidates = load(source['library_ref']['file'])['candidates']
        scopes = reconstruct_metadata(selected, candidates, source_path, budget)
        metadata_ref = args.output_dir / 'metadata_rows.json'
    args.output_dir.mkdir(parents=True)
    if prior is None:
        save(metadata_ref, scopes)
    roster = [(si, ai) for si, ai, _ in PILOT] if args.stage == 'pilot' else list(recovered)
    reuse_ids = {case['case_id'] for case in reused}
    cases, failures = [], list(preflight)
    for si, ai in roster:
        case_id = f'{si}:{ai}'
        if case_id in reuse_ids:
            old = next(case for case in reused if case['case_id'] == case_id)
            cases.append({'case_id': case_id, 'reused_from': reference(args.reuse_pilot,
                'cases', reused.index(old)), 'new_compute_cost': 0,
                'observed_outcome': old['observed_outcome'],
                'parameter_identifiability': old['parameter_identifiability'],
                'ridge_training_rejection_removed': old['ridge_training_rejection_removed'],
                'ridge_both_splits_rejection_removed': old['ridge_both_splits_rejection_removed']})
            continue
        if (si, ai) not in recovered:
            continue
        try:
            matrix = args.output_dir / f'matrix_{si}_{ai}.npz' if args.stage == 'pilot' else None
            cases.append(audit_case(si, ai, selected[si], scopes[str(si)], recovered[si, ai],
                                    source_path, budget, matrix))
        except MissingFixedStructure as error:
            failures.append({'case_id': case_id, 'status': 'UNKNOWN', 'reason': str(error)})
    output = {'kind': 'fixed_retained_parameter_fit_numerical_audit', 'stage': args.stage,
        'provenance': provenance, 'metadata_rows_ref': str(metadata_ref.resolve()),
        'pilot_selection': [{'search_index': si, 'attempt_index': ai, 'purpose': label}
                             for si, ai, label in PILOT],
        'preflight': {'saved_fit_calls': 723, 'recovered_free_slot_sets': len(recovered),
            'missing': len(preflight), 'factory_call_or_candidate_generation': False},
        'cases': cases, 'failures': failures,
        'summary': {'case_count': len(cases), 'reused_pilot_count': len(reused),
            'new_fit_calls': sum('reused_from' not in case for case in cases),
            'observed_outcomes': dict(Counter(c['observed_outcome'] for c in cases)),
            'parameter_identifiability': dict(Counter(c['parameter_identifiability'] for c in cases)),
            'ridge_training_rejections_removed': sum(c['ridge_training_rejection_removed'] for c in cases),
            'ridge_both_splits_rejections_removed': sum(c['ridge_both_splits_rejection_removed'] for c in cases)},
        'new_compute_cost': budget.spent, 'compute_budget': budget.limit,
        'metadata_matching_compute_included': prior is None,
        'wall_seconds': monotonic() - started,
        'old_cost_refs': [reference(source_path, 'additional_development_costs'),
                          reference(source_path.parent / 'comparison.json', 'accounting')],
        'new_observations': 0, 'provider_calls': 0, 'world_model_fits': 0,
        'scientific_support_added': False, 'independent_confirmation': False,
        'scope': 'Same retained before/after parameter targets, training/held split and fixed structures. Rank identification, actual train/held outcomes and linear-span attainability remain distinct. This audit never changes the old study or theory library.'}
    save(args.output_dir / 'audit.json', output)
    print(json.dumps({'audit_ref': str(args.output_dir / 'audit.json'),
                      'preflight': output['preflight'], 'summary': output['summary'],
                      'new_compute_cost': budget.spent, 'wall_seconds': output['wall_seconds']}))
    return output


if __name__ == '__main__':
    main()
