#!/usr/bin/env python3
"""Prepare registered studies or execute the declared saved-scene diagnostic.

The default source/successor stages only prepare. The explicit saved-scene-run
stage executes its separately registered development diagnostic. Other Python
callers may use ``run_prepared_source(output)`` after execution authorization.
The successor's learned library cannot be frozen until its source study finishes.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from random import Random
import sys
from time import monotonic
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    write_json_checkpoint(path, value, ensure_ascii=False)


def _value(reference):
    value = _read(reference['file'])
    for key in reference.get('json_path', []):
        value = value[key]
    return value


def _model_names(value):
    return [item if isinstance(item, str) else item['model'] for item in value]


def _public_question(template, fitting, namespace, phase, case_id):
    question = deepcopy(template)
    context = deepcopy(question['theory_target']['context_ref'])
    context['source_id'] = f'{namespace}/{phase}/{case_id}'

    def rebind(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'context_ref':
                    value[key] = deepcopy(context)
                else:
                    rebind(item)
        elif isinstance(value, list):
            for item in value:
                rebind(item)

    rebind(question)
    question['theory_target']['revision_fitting'] = deepcopy(fitting)
    panels = question['resolution_experiments']
    original_query = deepcopy(question['query_semantics']['query_expression'])
    matches = []
    for panel in panels:
        query = panel['query']
        if query == original_query:
            matches.append(panel['id'])
        if query.get('op') != 'Readout' or len(panel['actions']) != 1:
            raise ValueError('the existing public contract requires one Readout action per panel')
        query['record_index'] = {'experiment_id': panel['id'], 'action_index': 0}
    if len(matches) != 1:
        raise ValueError('the task readout must identify exactly one original public panel')
    question['query_semantics']['query_expression']['record_index'] = {
        'experiment_id': matches[0], 'action_index': 0}
    return question


def _inventory(spec):
    """Read declared metadata and stat model files; never export their contents."""
    from hegel_machine.cognition.code_agent import python_model_context

    materials = spec['materials']
    archive_path = Path(materials['baseline_archive']).resolve()
    archive = _read(archive_path)
    if 'files' in archive or not archive.get('workspace'):
        raise ValueError('prepare requires the small workspace archive metadata, not an expanded file view')
    workspace = Path(archive['workspace']).resolve()
    names = list(archive.get('research_roles', {}).get('predictors', []))
    roles = archive.get('research_roles', {})
    if roles.get('predictor'):
        names.append(roles['predictor'])
    names.extend(archive.get('finite_native_author', {}).get('predictor_paths', []))
    catalogs = list(archive.get('source_catalog', []))
    catalog_path = workspace / 'source_catalog.json'
    if catalog_path.is_file():
        catalogs.extend(_read(catalog_path))
    for item in catalogs:
        if item.get('model'):
            names.append(item['model'])
        names.extend(item.get('predictors', []))
    names.extend(_model_names(_value(materials['native_models_ref'])))
    names = list(dict.fromkeys(names))
    entries, batches = [], []

    def append(model, source, identifier, destination):
        path = source / model
        stat = path.stat()
        context, inner = python_model_context(source, model)
        entries.append({'id': identifier, 'workspace': str(source), 'model': model,
            'context': str(context), 'inner_model': inner, 'bytes': stat.st_size,
            'destination': destination,
            'destination_context': str(Path(destination).parents[len(Path(inner).parts)-1])})

    for model in names:
        append(model, workspace, 'baseline:' + model, model)
    for batch in materials.get('extra_predictor_batches', []):
        source = Path(batch['workspace']).resolve()
        models = sorted(str(path.relative_to(source)) for path in source.glob(batch['pattern'])
                        if path.is_file())
        if len(models) != batch['expected_count']:
            raise ValueError(f"{batch['id']}: expected {batch['expected_count']} models, found {len(models)}")
        # The same filename in another execution context remains another model.
        for model in models:
            append(model, source, batch['id'] + ':' + model,
                   str(Path('retained_batches') / batch['id'] / model))
        batches.append({**deepcopy(batch), 'model_count': len(models)})
    regression_refs = [{'file': str(Path(path).resolve()), 'json_path': []}
                       for path in materials['regression_refs']]
    for reference in regression_refs:
        Path(reference['file']).stat()
    return {'baseline_archive_ref': {'file': str(archive_path), 'json_path': []},
        'baseline_workspace': str(workspace), 'baseline_model_count': len(names),
        'model_count': len(entries), 'models': entries, 'extra_batches': batches,
        'regression_evidence_refs': regression_refs,
        'context_materialization': 'pending_full_context_export',
        'native_models_ref': deepcopy(materials['native_models_ref']),
        'source_failures': 'retained_in_original_archive_and_batch_evidence_refs',
        'new_observations': 0, 'new_predictions': 0, 'new_fits': 0, 'new_model_calls': 0}


def prepare(spec, output):
    """Save small plans, references and the finite author; acquire no labels."""
    from hegel_machine.cognition.finite_native_author import export_finite_native_author
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
    from scripts import run_anonymous_revision_inquiry as study

    spec_ref = None
    if not isinstance(spec, dict):
        spec_ref = {'file': str(Path(spec).resolve()), 'json_path': []}
        spec = _read(spec)
    spec = deepcopy(spec)
    output = Path(output).resolve()
    if (output / 'registration.json').exists():
        raise ValueError('use a fresh preparation directory')
    inventory = _inventory(spec)
    materials = spec['materials']
    template = _value(materials['public_question_ref'])
    fitting = _value(materials['revision_fitting_ref'])
    plans = {}
    for phase in ('source', 'target'):
        settings = spec[phase]
        limits = InquiryLLMConfig(**settings['limits'])
        if limits.max_model_calls != 0 or limits.token_budget != 0:
            raise ValueError('the finite source/successor protocol declares zero model calls and tokens')
        organization = {key: deepcopy(value) for key, value in spec['organization'].items()
                        if key != 'consolidation'}
        organization.update(phase=phase, theories=[],
            initial_library_path=deepcopy(materials['initial_library_ref']))
        if phase == 'source':
            organization['consolidation'] = deepcopy(spec['organization']['consolidation'])
        else:
            organization['pr3_library_path'] = spec['future_targets']['frozen_library_path']
        plan = study.protocol(kinds=settings['kinds'], cases=settings['cases'],
            seed=settings['seed'], arms=study.ORGANIZATION_ARMS, config=limits,
            research_organization=organization, author=spec['author'],
            archive=materials['baseline_archive'])
        plan['runtime_snapshot'] = sorted(set(plan['runtime_snapshot']) |
            {'scripts/prepare_pr3_source_successor.py'})
        plan['prepared_public_questions'] = {case['id']: _public_question(
            template, fitting, spec['namespace'], phase, case['id']) for case in plan['cases']}
        plans[phase] = plan
    source_seeds = {case['seed'] for case in plans['source']['cases']}
    if source_seeds.intersection(case['seed'] for case in plans['target']['cases']):
        raise ValueError('registered source and successor seeds must differ')
    driver = output / spec['author']['path']
    driver.parent.mkdir(parents=True, exist_ok=True)
    driver.write_text(export_finite_native_author(), encoding='utf-8')
    for phase, plan in plans.items():
        _write(output / f'{phase}_plan.json', plan)
    _write(output / 'baseline_inventory.json', inventory)
    registration = {'status': 'prepared_not_executed', 'spec_ref': spec_ref, 'spec': spec,
        'plans': {phase: {'file': str(output / f'{phase}_plan.json'), 'json_path': []}
                  for phase in plans},
        'baseline_inventory_ref': {'file': str(output / 'baseline_inventory.json'), 'json_path': []},
        'author_ref': {'file': str(driver)}, 'baseline_model_count': inventory['model_count'],
        'baseline_materialization': inventory['context_materialization'],
        'target_library_status': 'pending_source_result_and_pre_target_freeze',
        'runtime': 'run_study_will_freeze_current_mainline_at_authorized_source_launch',
        'scientific_scope': 'New seeds do not establish new structure or an independent mechanism. '
                            'Old source revisions remain development/regression evidence.',
        'new_observations': 0, 'new_predictions': 0, 'new_fits': 0, 'new_model_calls': 0,
        'new_compute_cost': 0}
    _write(output / 'registration.json', registration)
    return registration


def _frozen_proposal_qualification(identifier, candidate, declared_sources):
    """Admit a typed local proposal, without claiming world-law support."""
    from hegel_machine.cognition.language import Definition, Program

    reasons = []
    reconstruction = candidate.get('source_reconstruction', [])
    sources = sorted(set(candidate.get('source_ids', [])).intersection(declared_sources,
        {row.get('source_id') for row in reconstruction}))
    if len(sources) < 2:
        reasons.append('two_registered_physical_sources_required')
    if not reconstruction or not all(row.get('exact') is True for row in reconstruction):
        reasons.append('exact_source_reconstruction_not_established')
    if candidate.get('kind') != 'program_revision_relation':
        reasons.append('executable_revision_proposal_required')
    if candidate.get('purpose') not in {'structural_revision', 'parameter_proposal', 'exact_edit_map'}:
        reasons.append('proposal_purpose_required')
    if (not candidate.get('output_unit') or not candidate.get('fixed_role_signature')
            or any(not hole.get('output_unit') or not hole.get('time_contract')
                   or not hole.get('value_type') for hole in candidate.get('holes', []))):
        reasons.append('declared_roles_units_and_time_required')
    try:
        Definition.from_dict(candidate['definition'])
        Program.from_dict(candidate['before_pattern'])
    except (KeyError, TypeError, ValueError):
        reasons.append('saved_executable_structure_unavailable')
    return {'id': identifier, 'version': candidate.get('version', 1),
        'status': 'qualified_local_proposal' if not reasons else 'UNKNOWN',
        'purpose': candidate.get('purpose'), 'physical_source_ids': sources,
        'reasons': reasons, 'source_world_success_inherited': False,
        'old_closure_reconstructibility_is_admission_requirement': False}


def _successor_completion_budget(models, spec):
    """Reuse the ordinary completion hold on maximum registered paid rows."""
    from hegel_machine.cognition.finite_native_author import ordinary_completion_reserve

    settings = spec['target']['limits']
    agent = SimpleNamespace(knowledge={'finite_native_author': {'predictor_paths': models}},
        operator_results=[], candidate_paths=[], rows=[None] * settings['observation_budget'],
        evaluated_model={'evaluated_rows': 0}, _inquiry_stack=[None],
        config=SimpleNamespace(confirmation_rows=settings['confirmation_rows']))
    hold = ordinary_completion_reserve(agent, {'theory_target': {'revision_fitting': {}}})
    routing = spec['organization'].get('routing_allocation', {}).get('compute', 0)
    native = spec['native_search']['max_compute']
    required = hold['compute_cost'] + routing + native
    return {'completion_reserve': hold, 'common_routing_compute': routing,
        'common_native_search_compute': native, 'required_per_case_compute': required,
        'registered_per_case_compute': settings['compute_budget'],
        'registered_cap_sufficient': settings['compute_budget'] >= required,
        'scope': 'One complete maximum-row common-pool comparison and finish, one stale incumbent, one parent delivery, common routing and one native allowance. Not a guarantee for all48 author steps.'}


def _direct_candidate(entry):
    """Read provenance used by the existing transfer and index consumers."""
    for field in ('consolidation_ref', 'binding', 'request_basis'):
        value = entry.get(field) or {}
        if value.get('candidate_id'):
            return value['candidate_id']
    return entry.get('family_id')


def _frozen_entry_ref(reference, library, field, key, *parts):
    from hegel_machine.cognition.theory_consolidation import library_entry_reference

    proxy = SimpleNamespace(workspace=ROOT, theory_consolidation=library,
        _theory_library_base=library, _theory_library_source_ref=reference)
    return library_entry_reference(proxy, field, key, *parts)


def _index_entry_ref(reference, identifier, *parts):
    reference = deepcopy(reference)
    while True:
        document = _value(reference)
        if 'source_ref' not in document or 'entries' not in document:
            return {**reference, 'json_path': [*reference.get('json_path', []), identifier, *parts]}
        if identifier in document['entries']:
            return {**reference, 'json_path': [*reference.get('json_path', []), 'entries', identifier, *parts]}
        reference = deepcopy(document['source_ref'])


def _freeze_removal(plan, library_ref, library, baseline, index_ref, indexed,
                    identifiers, output, kind, *, class_ref=None):
    """Freeze an automatic-organization intervention, keeping concrete facts."""
    old = set(baseline.get('candidates', {}))
    removed = set(identifiers)
    if not removed or not removed <= set(library['candidates']) or removed.intersection(old):
        raise ValueError('remove only registered new candidates; retain every old candidate')
    directory = Path(output) / 'interventions' / kind
    library_path, manifest_path = directory / 'theory_library.json', directory / 'manifest.json'
    direct_executions = [key for key, entry in library.get('executions', {}).items()
                         if _direct_candidate(entry) in removed]
    transfers = [index for index, entry in enumerate(library.get('transfers', []))
                 if _direct_candidate(entry) in removed]
    removed_index, overrides, removed_versions = [], {}, []
    for identifier, entry in indexed.items():
        versions = [*entry.get('history', []), {key: value for key, value in entry.items() if key != 'history'}]
        surviving = []
        for position, version in enumerate(versions):
            if _direct_candidate(version) not in removed:
                surviving.append(deepcopy(version))
                continue
            removed_versions.append({'id': identifier, 'version': version.get('version', 1),
                'candidate_id': _direct_candidate(version), 'source_ref': _index_entry_ref(
                    index_ref, identifier, *(['history', position] if position < len(versions) - 1 else []))})
        if len(surviving) == len(versions):
            continue
        if not surviving:
            removed_index.append(identifier)
        else:
            restored = surviving.pop()
            if surviving:
                restored['history'] = surviving
            overrides[identifier] = restored
    # Mixed source-construction cache records are retained history. The target
    # caller does not consolidate source Episodes, and these cached results only
    # return IDs; they do not reinstall a removed candidate into the library.
    historical = []
    for key, execution in library.get('executions', {}).items():
        if key in direct_executions:
            continue
        for field in ('frozen_candidates', 'parameter_relation_searches'):
            for position, entry in enumerate(execution.get('result', {}).get(field, [])):
                if entry.get('candidate_id') in removed:
                    historical.append({'candidate_id': entry['candidate_id'], 'source_ref':
                        _frozen_entry_ref(library_ref, library, 'executions', key, 'result', field, position)})
    view = {'source_ref': deepcopy(library_ref), 'removed_candidates': sorted(removed)}
    if direct_executions:
        view['removed_execution_keys'] = direct_executions
    if transfers:
        view['removed_transfer_indices'] = transfers
    _write(library_path, view)
    intervention = deepcopy(plan)
    intervention['research_organization']['pr3_library_path'] = str(library_path)
    intervention['target_intervention'] = {'kind': kind,
        'manifest_ref': {'file': str(manifest_path), 'json_path': []}}
    index_view = None
    if removed_index or overrides:
        if index_ref is None:
            raise ValueError('derived index removal needs its original source reference')
        index_view = {'source_ref': deepcopy(index_ref), 'entries': overrides,
                      'removed_entries': removed_index}
        index_path = directory / 'theory_index.json'
        _write(index_path, index_view)
        intervention['target_intervention']['index_view_ref'] = {'file': str(index_path), 'json_path': []}
    manifest = {'status': 'frozen_ready', 'kind': kind, 'selected_class_ref': deepcopy(class_ref),
        'source_library_ref': deepcopy(library_ref), 'source_index_ref': deepcopy(index_ref),
        'removed_candidate_ids': sorted(removed), 'preserved_old_candidate_ids': sorted(old),
        'candidate_alias_refs': [_frozen_entry_ref(library_ref, library, 'candidates', identifier)
                                 for identifier in sorted(removed)],
        'remaining_candidate_ids': [key for key in library['candidates'] if key not in removed],
        'removed_execution_keys': direct_executions, 'removed_transfer_indices': transfers,
        'removed_index_ids': removed_index, 'restored_independent_index_ids': list(overrides),
        'removed_index_versions': removed_versions,
        'retained_historical_construction_refs': historical,
        'consumer_inventory': {'candidate_count': len(library['candidates']),
            'index_entry_count': len(indexed), 'execution_count': len(library.get('executions', {})),
            'transfer_count': len(library.get('transfers', [])),
            'direct_provenance_fields': ['consolidation_ref.candidate_id', 'binding.candidate_id',
                                        'request_basis.candidate_id', 'family_id']},
        'automatic_policy_change': 'Use filtered frozen library/index only; target has no automatic source consolidation.',
        'automatic_library_view_ref': {'file': str(library_path), 'json_path': []},
        'automatic_index_view_ref': deepcopy(intervention['target_intervention'].get('index_view_ref')),
        'source_cost_ref': deepcopy(plan['source_cost_ref']),
        'concrete_source_programs_parameters_facts_removed': False,
        'old_language_and_independent_old_reconstruction_legal': True,
        'old_closure_reconstructibility_is_veto': False,
        'source_failures_costs_and_history_preserved_by_reference': True}
    _write(manifest_path, manifest)
    _write(directory / 'plan.json', intervention)
    return {'status': 'prepared_not_executed',
        'plan_ref': {'file': str(directory / 'plan.json'), 'json_path': []},
        'manifest_ref': {'file': str(manifest_path), 'json_path': []},
        'frozen_remove_library_ref': {'file': str(library_path), 'json_path': []},
        'removed_candidate_ids': sorted(removed)}, view, index_view


def prepare_frozen_successor(spec, output):
    """Freeze reviewable successor inputs from existing source results only.

    No library induction, archive materialization or laboratory is performed.
    Ordinary code/native fitting and the original confirmation contract remain
    the consumers of these local proposals on a separately authorized target.
    """
    from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
    from hegel_machine.cognition.finite_native_author import export_finite_native_author
    from hegel_machine.cognition.theory_consolidation import _read_library, _read_index
    from scripts import run_anonymous_revision_inquiry as study

    spec_ref = None
    if not isinstance(spec, dict):
        spec_ref = {'file': str(Path(spec).resolve()), 'json_path': []}
        spec = _read(spec)
    spec = deepcopy(spec)
    output = Path(output).resolve()
    if (output / 'registration.json').exists():
        raise ValueError('use a fresh preparation directory')
    frozen = spec['frozen_source']
    # Read small frozen statements and the existing library, not source states,
    # training snapshots or paid trajectories.
    manifest = _value(frozen['manifest_ref'])
    source_plan = _value(frozen['source_plan_ref'])
    accounting = _value(frozen['accounting_ref'])
    if not manifest or not accounting['accounting_and_preservation_checks_pass']:
        raise ValueError('closed source freeze and accounting are required')
    initial_ref = spec['materials']['initial_library_ref']
    library_ref = frozen['library_ref']
    ids = frozen['new_candidate_ids']
    # A no-object successor uses the identical fixed library in B and C.
    # Identity is already explicit in this contract; expanding its complete
    # execution history twice adds no qualification or preservation evidence.
    same_fixed_library = not ids and library_ref == initial_ref
    if not same_fixed_library:
        baseline = _read_library(initial_ref, ROOT)
        library = _read_library(library_ref, ROOT)
    declared = {question['theory_target']['context_ref']['source_id']
                for question in source_plan['prepared_public_questions'].values()}
    if not same_fixed_library and set(ids).intersection(baseline.get('candidates', {})):
        raise ValueError('new proposal IDs must retain a distinct current-source identity')
    if not same_fixed_library and any(library.get('candidates', {}).get(identifier) != candidate
           for identifier, candidate in baseline.get('candidates', {}).items()):
        raise ValueError('the frozen source library must retain the complete common old library')
    qualification = []
    for identifier in ids:
        row = _frozen_proposal_qualification(identifier, library['candidates'][identifier], declared)
        row['candidate_ref'] = _frozen_entry_ref(library_ref, library, 'candidates', identifier)
        qualification.append(row)
    qualified = sorted(row['id'] for row in qualification if row['status'] == 'qualified_local_proposal')
    unavailable = ([] if same_fixed_library else [identifier for identifier in library.get('candidates', {})
                   if identifier not in baseline.get('candidates', {}) and identifier not in qualified])
    selected = qualified[0] if qualified else None
    keep_path = output / 'frozen_target_libraries' / 'keep.json'
    keep = ({'source_ref': deepcopy(library_ref), 'removed_candidates': unavailable}
            if qualified else {'source_ref': deepcopy(initial_ref)})
    settings = spec['target']
    limits = InquiryLLMConfig(**settings['limits'])
    if limits.max_model_calls != 0 or limits.token_budget != 0:
        raise ValueError('the finite successor protocol declares zero model calls and tokens')
    organization = {key: deepcopy(value) for key, value in spec['organization'].items()
                    if key != 'consolidation'}
    organization.update(phase='target', theories=[], initial_library_path=deepcopy(initial_ref),
        pr3_library_path=str(keep_path))
    plan = study.protocol(kinds=settings['kinds'], cases=settings['cases'], seed=settings['seed'],
        arms=study.ORGANIZATION_ARMS, config=limits, research_organization=organization,
        author=spec['author'], archive=spec['materials']['baseline_archive'])
    # This entry supplies the real prefix and C-index callbacks at execution.
    plan['runtime_snapshot'] = sorted(set(plan['runtime_snapshot']) |
        {'scripts/prepare_pr3_source_successor.py'})
    if {case['seed'] for case in source_plan['cases']}.intersection(case['seed'] for case in plan['cases']):
        raise ValueError('registered source and successor seeds must differ')
    template = _value(spec['materials']['public_question_ref'])
    fitting = _value(spec['materials']['revision_fitting_ref'])
    plan['prepared_public_questions'] = {case['id']: _public_question(
        template, fitting, spec['namespace'], 'target', case['id']) for case in plan['cases']}
    plan['source_cost_ref'] = deepcopy(frozen['accounting_ref'])
    readiness = spec['design'].get('target_seed_readiness', {})
    target_seeds = [case['seed'] for case in plan['cases']]
    if readiness:
        original_target = _value(readiness['original_unexecuted_target_plan_ref'])
        if target_seeds != readiness['expected_target_seeds'] or target_seeds != [
                case['seed'] for case in original_target['cases']]:
            raise ValueError('successor preparation must retain the registered unused target seeds')
        target_paths = [{'path': str(Path(path)), 'exists': Path(path).exists()}
                        for path in readiness['unconsumed_target_directories_to_check']]
        if any(row['exists'] for row in target_paths):
            raise ValueError('a registered target directory already exists; unused target scope is not established')
    else:
        target_paths = []
    _write(keep_path, keep)
    _write(output / 'target_plan.json', plan)
    author_path = output / spec['author']['path']
    author_path.parent.mkdir(parents=True, exist_ok=True)
    author_path.write_text(export_finite_native_author(), encoding='utf-8')
    individual = {'status': 'not_applicable_no_qualified_new_object', 'selected_candidate_id': None}
    new_class = {'status': 'not_applicable_no_qualified_new_object',
                 'ordinary_old_language_reconstruction_remains_legal': True}
    if selected:
        index_ref = frozen.get('index_ref')
        indexed = _read_index(index_ref, ROOT) if index_ref else {}
        eligible_library = deepcopy(library)
        for identifier in unavailable:
            eligible_library['candidates'].pop(identifier)
        eligible_ref = {'file': str(keep_path), 'json_path': []}
        individual, individual_view, individual_index = _freeze_removal(plan, eligible_ref,
            eligible_library, baseline, index_ref, indexed, [selected], output, 'individual_object')
        individual.update(selected_candidate_id=selected,
            selection_rule='first_lexicographic_qualified_current_source_proposal',
            matched_cases_ref={'file': str(output / 'target_plan.json'), 'json_path': ['cases']},
            matched_limits_ref={'file': str(output / 'target_plan.json'), 'json_path': ['limits']})
        groups_ref = frozen.get('redundancy_classes_ref')
        groups = _value(groups_ref) if groups_ref else []
        selected_group = next(((position, group) for position, group in enumerate(groups)
            if selected in [*group.get('original_member_ids', []),
                            *group.get('additional_same_contract_member_ids', [])]), None)
        if selected_group is None:
            new_class = {'status': 'UNKNOWN', 'reason': 'retained_class_membership_required',
                'ordinary_old_language_reconstruction_remains_legal': True}
        else:
            position, group = selected_group
            aliases = [*group.get('original_member_ids', []),
                       *group.get('additional_same_contract_member_ids', [])]
            if not set(aliases) <= set(qualified):
                raise ValueError('the selected class must reference qualified frozen new aliases')
            class_ref = {**deepcopy(groups_ref), 'json_path': [*groups_ref.get('json_path', []), position]}
            new_class, class_view, class_index = _freeze_removal(plan, eligible_ref,
                eligible_library, baseline, index_ref, indexed, aliases, output,
                'new_redundancy_class', class_ref=class_ref)
            new_class.update(selected_candidate_id=selected, selected_class_index=position,
                ordinary_old_language_reconstruction_remains_legal=True,
                registered_class_groups_ref=deepcopy(groups_ref))
            if class_view == individual_view and class_index == individual_index:
                new_class['reuse_execution_of'] = 'individual_object'
    # Count-only names call the production reserve contract without reading or
    # running predictors. Actual common archive size is checked before launch.
    estimate = spec['design'].get('common_pool_model_count_upper_estimate', 999)
    budget = _successor_completion_budget([f'count_only/{i}' for i in range(estimate)], spec)
    common_archive = ({'entry': 'hegel_machine.cognition.anonymous_research_archive.resolve_workspace_archive',
        'archive_ref': deepcopy(frozen['common_archive_ref']),
        'scope': 'The prepared common archive retains every explicit predictor at its original execution context; all A/B/C receive the same facts, failures and paid references.',
        'status': 'prepared_common_archive_referenced'} if frozen.get('common_archive_ref') else
        {'entry': 'hegel_machine.cognition.anonymous_research_archive.prepare_frozen_source_archive',
        'source_study': frozen['source_study'], 'base_archive_ref': deepcopy(frozen['common_base_archive_ref']),
        'scope': 'All registered source contexts, concrete versions, parameters, failures and paid references are common to A/B/C.',
        'status': 'not_materialized_preparation_only'})
    registration = {'status': 'prepared_not_executed', 'stage': 'frozen_successor',
        'namespace': spec['namespace'], 'spec_ref': spec_ref, 'spec': spec,
        'author_ref': {'file': str(author_path)},
        'source_freeze_ref': deepcopy(frozen['manifest_ref']),
        'source_plan_ref': deepcopy(frozen['source_plan_ref']),
        'source_summary_ref': deepcopy(frozen['source_summary_ref']),
        'closed_source_accounting_ref': deepcopy(frozen['accounting_ref']),
        'proposal_qualification': qualification, 'qualified_local_proposal_count': len(qualified),
        'automatic_C_library_ref': {'file': str(keep_path), 'json_path': []},
        'whole_strategy': {'status': 'prepared_not_executed', 'arms': list(study.ORGANIZATION_ARMS),
            'plan_ref': {'file': str(output / 'target_plan.json'), 'json_path': []},
            'source_failure_denominator_preserved': True, 'no_new_object_C_falls_back_to_B': not qualified},
        'individual_object': individual,
        'new_class_intervention': new_class,
        'old_closure_reconstruction_diagnostic_ref': deepcopy(frozen['reconstruction_diagnostic_ref']),
        'reconstruction_blocks_successor': False,
        'common_archive_preparation': common_archive,
        'budget_readiness': {'status': 'pending_actual_common_pool_count_and_frozen_equal_cap',
            'count_only_estimate': estimate, 'estimate': budget,
            'actual_common_model_count': None,
            'shared_by': ['archive', 'fixed_pr2', 'pr3', 'individual_remove', 'new_class_remove'],
            'insufficient_actual_budget_action': 'Stop before run_study and re-register equal budgets; preserve all models and costs.'},
        'target_use': {'structural_revision': 'Conditional structure/research proposal followed by ordinary target fitting and original prediction/confirmation checks.',
            'parameter_proposal': 'Saved parameter-search proposal followed by ordinary whole-after target fitting. Numerical initialization quality remains unevaluated; exact historical after coefficients are not required.',
            'exact_edit_map': 'Sufficient-condition special case under its existing role/input and check contracts.'},
        'estimand': deepcopy(spec['design']['endpoints']),
        'shared_algorithm_contract': {'same_numerical_units_scheduling_and_tools': True,
            'shared_scheduler_improvement_is_not_PR3_object_effect': True,
            'equivalent_old_program_reconstruction_is_valid_baseline_success': True},
        'target_seed_readiness': {'seeds': target_seeds,
            'registered_target_directories': target_paths,
            'target_labels_generated_or_read': False},
        'runtime': 'fresh_current_mainline_freeze_required_at_separately_authorized_launch',
        'successor_execution_authorized': False,
        'new_compute_cost': 0, 'new_observations': 0, 'new_predictions': 0, 'new_fits': 0, 'new_model_calls': 0}
    _write(output / 'registration.json', registration)
    return registration


def _source_knowledge(registration, inventory):
    """Materialize static executable contexts and evidence refs, without research."""
    from hegel_machine.cognition.anonymous_research_archive import resolve_workspace_archive
    from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent

    spec = registration['spec']
    archive = _read(spec['materials']['baseline_archive'])
    base = [entry['model'] for entry in inventory['models'] if entry['id'].startswith('baseline:')]
    archive.setdefault('research_roles', {})['predictors'] = base
    knowledge = resolve_workspace_archive(archive, training={})
    files = knowledge['files']
    paths, contexts = list(base), {}
    for entry in inventory['models']:
        if entry['id'].startswith('baseline:'):
            continue
        context = Path(entry['context'])
        prefix = Path(entry['destination_context'])
        key = str(context), str(prefix)
        contexts.setdefault(key, []).append(entry)
    catalog = knowledge.setdefault('source_catalog', [])
    for (context, prefix), entries in contexts.items():
        models = [entry['inner_model'] for entry in entries]
        view = SimpleNamespace(workspace=Path(context), files={},
            knowledge={'research_roles': {'predictors': models}})
        exported = CommonArchiveInquiryAgent._result_files(view, {}, [], models)
        for name, content in exported.items():
            destination = str(Path(prefix) / name)
            if destination in files:
                raise ValueError(f'additional execution context would overwrite {destination}')
            files[destination] = content
        paths.extend(entry['destination'] for entry in entries)
        catalog.append({'id': entries[0]['id'].split(':', 1)[0], 'directory': prefix,
            'predictors': [entry['destination'] for entry in entries],
            'model_sources': [{'model': entry['destination'], 'source_model': entry['model'],
                              'source_ref': {'file': str(Path(entry['workspace']) / entry['model'])}}
                             for entry in entries],
            'evidence_refs': [batch['evidence_ref'] for batch in inventory['extra_batches']
                              if batch['id'] == entries[0]['id'].split(':', 1)[0]]})
    missing = [path for path in paths if path not in files]
    if missing:
        raise ValueError(f'complete baseline materialization missing: {missing[:3]}')
    driver = registration['author_ref']['file']
    files[spec['author']['path']] = Path(driver).read_text(encoding='utf-8')
    files['source_catalog.json'] = json.dumps(catalog, ensure_ascii=False)
    files['shared/retained_predictor_catalog.json'] = json.dumps({'predictors': paths,
        'inventory_ref': registration['baseline_inventory_ref']}, ensure_ascii=False)
    knowledge['research_roles']['predictors'] = paths
    knowledge['finite_native_author'] = {**spec['native_search'], 'predictor_paths': paths,
        'predictor_catalog_ref': {'file': 'shared/retained_predictor_catalog.json', 'json_path': []}}
    knowledge['regression_evidence_refs'] = deepcopy(inventory['regression_evidence_refs'])
    files['shared/regression_evidence_refs.json'] = json.dumps({
        'scope': 'Common old diagnostics/libraries, not new source evidence or automatic support.',
        'sources': knowledge['regression_evidence_refs']}, ensure_ascii=False)
    knowledge['archive_instructions'] = knowledge.get('archive_instructions', '') + (
        '\nshared/regression_evidence_refs.json addresses the shared prior diagnostics and libraries. '
        'Ordinary Python can read these JSON files; existing load_source_values and library '
        'readers accept their absolute file/json_path references. Large old libraries remain at '
        'their original locations; the automatic initial library is selected separately.')
    return knowledge


def run_prepared_source(output, *, workers=8):
    """Run the existing source study after authorization; the CLI never calls it."""
    from scripts import run_anonymous_revision_inquiry as study

    output = Path(output).resolve()
    registration = _read(output / 'registration.json')
    plan = _read(output / 'source_plan.json')
    inventory = _read(output / 'baseline_inventory.json')
    knowledge = _source_knowledge(registration, inventory)
    return _run_prepared_plan(plan, output / 'source', knowledge, workers=workers)


def run_prepared_successor(output, *, workers=8):
    """Use the original source-context exporter and current formal target loop.

    CLI preparation never calls this function. Actual successor execution has
    its own authorization boundary, independent of source/object qualification.
    """
    from hegel_machine.cognition.anonymous_research_archive import (
        prepare_frozen_source_archive, resolve_workspace_archive)

    output = Path(output).resolve()
    registration = _read(output / 'registration.json')
    if not registration['successor_execution_authorized']:
        raise ValueError('successor execution is not authorized by this prepared registration')
    spec = registration['spec']
    common = registration['common_archive_preparation']
    knowledge = (resolve_workspace_archive(_value(common['archive_ref']))
        if common.get('archive_ref') else
        prepare_frozen_source_archive(common['source_study'], common['base_archive_ref']['file']))
    predictors = list(dict.fromkeys(knowledge['research_roles']['predictors']))
    knowledge['files'][spec['author']['path']] = Path(registration['author_ref']['file']).read_text(encoding='utf-8')
    knowledge['finite_native_author'] = {**spec['native_search'], 'predictor_paths': predictors,
        'predictor_catalog_ref': {'file': 'source_catalog.json', 'json_path': []}}
    budget = _successor_completion_budget(predictors, spec)
    _write(output / 'launch_budget_preflight.json', {'status': ('sufficient' if budget['registered_cap_sufficient']
        else 'insufficient_no_scientific_execution'), 'actual_common_model_count': len(predictors), **budget})
    if not budget['registered_cap_sufficient']:
        raise ValueError(f'complete common pool requires at least {budget["required_per_case_compute"]} compute per case; re-register equal budgets before execution')
    plan = _read(output / 'target_plan.json')
    if plan['limits']['compute_budget'] != budget['registered_per_case_compute']:
        raise ValueError('the common target plan must carry the frozen equal compute budget')
    return _run_prepared_plan(plan, output / 'target', knowledge, workers=workers)


def _run_prepared_plan(plan, destination, knowledge, *, workers=8):
    """The existing paid-prefix and registered Common caller, shared by phases."""
    from scripts import run_anonymous_revision_inquiry as study

    agent_factory, prepare_prefix = _prepared_callbacks(plan)
    return study.run_study(plan, destination, knowledge=knowledge,
        prepare_prefix=prepare_prefix, agent_factory=agent_factory, workers=workers)


def _prepared_callbacks(plan):
    from scripts import run_anonymous_revision_inquiry as study

    def agent_factory(provider, workspace, **kwargs):
        case_id = Path(workspace).parent.name
        kwargs['knowledge']['finite_native_author']['question'] = deepcopy(
            plan['prepared_public_questions'][case_id])
        score_index = plan.get('paid_score_index_refs_by_case', {}).get(case_id)
        if score_index is not None:
            kwargs['knowledge'].setdefault('files', {})[
                '_paid_python_scores/_evaluated_model/assessments.json'] = json.dumps(_value(score_index))
        index_view = plan.get('target_intervention', {}).get('index_view_ref')
        if index_view and kwargs['arm'] == 'pr3':
            kwargs['knowledge']['theory_index'] = deepcopy(index_view)
        return study.RegisteredOrganizationAgent(provider, workspace, **kwargs)

    def prepare_prefix(agent, laboratory, case):
        actions = laboratory.confirmation_actions(plan['prefix_rows'], Random(case['prefix_seed']))
        feedback = agent._execute('query', {'actions': actions}, laboratory)
        if feedback.get('error'):
            raise ValueError(feedback['error'])
        agent.events.append({'step': -1, 'action': 'query', 'arguments': {'actions': actions},
            'feedback': feedback, 'usage': {'backend': 'common_paid_prefix', 'total_tokens': 0}})
        question = plan['prepared_public_questions'][case['id']]
        agent._open_question(deepcopy(question))
        agent._active_question_id = question['id']
        return {}

    return agent_factory, prepare_prefix


def run_prepared_intervention(output, kind, *, workers=8):
    """Run only the registered removal C, pairing it with the original keep C."""
    from scripts import run_anonymous_revision_inquiry as study

    output = Path(output).resolve()
    registration = _read(output / 'registration.json')
    if not registration['successor_execution_authorized']:
        raise ValueError('successor execution is not authorized by this prepared registration')
    field = {'individual_object': 'individual_object',
             'new_redundancy_class': 'new_class_intervention'}[kind]
    selected = registration[field]
    if selected['status'] != 'prepared_not_executed':
        return {'status': selected['status'], 'new_compute_cost': 0,
                'new_observations': 0, 'reason': selected.get('reason')}
    physical = selected.get('reuse_execution_of', kind)
    physical_field = {'individual_object': 'individual_object',
                      'new_redundancy_class': 'new_class_intervention'}[physical]
    prepared = registration[physical_field]
    destination = output / 'intervention_runs' / physical
    if physical != kind and (destination / 'summary.json').is_file():
        return {'status': 'reused_identical_intervention',
            'summary_ref': {'file': str(destination / 'summary.json'), 'json_path': []},
            'new_compute_cost': 0, 'new_observations': 0}
    plan = _value(prepared['plan_ref'])
    agent_factory, prepare_prefix = _prepared_callbacks(plan)
    result = study.run_target_intervention(plan, destination, keep_study=output / 'target',
        prepare_prefix=prepare_prefix, agent_factory=agent_factory, workers=workers)
    if physical != kind:
        return {'status': 'shared_execution_completed',
            'summary_ref': {'file': str(destination / 'summary.json'), 'json_path': []},
            'executed_for': physical, 'also_estimates': kind,
            'cost_accounting': 'Charge the referenced physical execution once.'}
    return result


def prepare_source_comparisons(source_study, output, compute_increment=150000):
    """Register one frozen-pool comparison per unresolved source, without running it.

    The increment extends the registered caps, retaining their unused balances.
    Completed submissions retain their original training and terminal references.
    """
    source = Path(source_study).resolve()
    output = Path(output).resolve()
    registration_path = output / 'comparison_registration.json'
    if registration_path.exists():
        raise ValueError('use a fresh comparison preparation directory')
    if type(compute_increment) is not int or compute_increment <= 0:
        raise ValueError('comparison compute increment must be a positive integer')
    original = _read(source / 'protocol.json')
    summary = _read(source / 'summary.json')
    if (original.get('author', {}).get('kind') != 'finite_native'
            or original.get('research_organization', {}).get('phase') != 'source'):
        raise ValueError('continue the registered finite-native source study')
    plan = deepcopy(original)
    plan['author_source_ref'] = {'file': str(source / 'initial_archive.json'),
        'json_path': ['files', plan['author']['path']]}
    plan['limits']['compute_budget'] += compute_increment
    cohort = plan['source_cohort_allocation']
    cohort['source_compute_limit'] += compute_increment * len(plan['cases'])
    for identifier in cohort['source_stage_compute']:
        cohort['source_stage_compute'][identifier] += compute_increment
    jobs = []
    for trial in summary['trials']:
        workspace = source / trial['case'] / trial['arm']
        old_cap = (original['source_cohort_allocation']['source_stage_compute'][trial['case']]
                   if trial['arm'] == 'pr3' else original['limits']['compute_budget'])
        jobs.append({'case': trial['case'], 'arm': trial['arm'],
            'mode': 'retained_completed' if trial['model'] else 'one_paid_comparison',
            'training_ref': {'file': str(workspace / 'training.json'), 'json_path': []},
            'evaluation_ref': {'file': str(workspace / 'evaluation.json'), 'json_path': []},
            'original_status': trial['status'], 'prior_costs': deepcopy(trial['costs']),
            'original_compute_cap': old_cap, 'compute_cap': old_cap + compute_increment})
    _write(output / 'comparison_plan.json', plan)
    registration = {'status': 'prepared_not_executed', 'source_study': str(source),
        'source_protocol_ref': {'file': str(source / 'protocol.json'), 'json_path': []},
        'source_summary_ref': {'file': str(source / 'summary.json'), 'json_path': []},
        'initial_archive_ref': {'file': str(source / 'initial_archive.json'), 'json_path': []},
        'plan_ref': {'file': str(output / 'comparison_plan.json'), 'json_path': []},
        'compute_increment_per_original_job': compute_increment, 'jobs': jobs,
        'execution': {'max_steps': 1, 'allow_confirmation': False, 'before_hook': False,
            'after_hook': 'RegisteredOrganizationAgent._after_model_turn',
            'candidate_pool': 'exact saved final research_choice roster',
            'new_observations': 0, 'new_world_predictor_generation': 0, 'new_model_calls': 0,
            'terminal_evaluation': 'original references only'},
        'scope': 'Same consumed historical sources and paid rows; not independent successors '
                 'or a complete rerun of the ABC research policies.'}
    _write(registration_path, registration)
    return registration


def _comparison_preflight(agent, laboratory, path, expected):
    """Read the actual finite author's next action before allowing its one step."""
    source = (agent.workspace / path).read_text(encoding='utf-8')
    namespace = {'__name__': 'retained_finite_author', '__file__': str(agent.workspace / path)}
    started, previous_cwd = monotonic(), Path.cwd()
    try:
        os.chdir(agent.workspace)
        exec(compile(source, str(agent.workspace / path), 'exec'), namespace)
        proposal = namespace['select_actions'](laboratory.public_spec(),
            deepcopy(agent.observations), agent._program_memory(path, {'path': path}))
    finally:
        os.chdir(previous_cwd)
        agent.costs['python_execution_seconds'] += monotonic() - started
    arguments = proposal.get('arguments', {}) if isinstance(proposal, dict) else {}
    return {'status': 'matched' if proposal == expected else 'comparison_not_executed',
        'action': proposal.get('action') if isinstance(proposal, dict) else None,
        'operator': arguments.get('operator'), 'model_count': len(arguments.get('models', [])),
        'saved_model_count': len(expected['arguments']['models']),
        'exact_saved_proposal': proposal == expected}


def run_prepared_source_comparisons(output):
    """Execute the registered comparison-only continuation after authorization.

    No organization before-hook, native search, acquisition, confirmation, or
    terminal evaluation runs. The original procedure dispatch and paid scorer
    perform the comparison and its actual retention decisions.
    """
    from dataclasses import replace
    from hegel_machine.cognition.code_agent import clone_or_copy
    from hegel_machine.cognition.temporal_paired_method import execute_procedure
    from scripts import run_anonymous_revision_inquiry as study

    output = Path(output).resolve()
    registration = _read(output / 'comparison_registration.json')
    source = Path(registration['source_study'])
    destination = output / 'source'
    if destination.exists():
        raise ValueError('retain the earlier continuation; use a fresh prepared directory')
    destination.mkdir(parents=True)
    plan = _value(registration['plan_ref'])
    organization, knowledge = study._prepare_organization(
        plan, destination, _value(registration['initial_archive_ref']), source)
    snapshot = sorted({str(path.relative_to(ROOT))
        for path in (ROOT / 'src/hegel_machine').rglob('*.py')} |
        {'scripts/run_anonymous_revision_inquiry.py', 'scripts/prepare_pr3_source_successor.py',
         'scripts/run_online_inquiry_llm.py', 'scripts/run_online_inquiry_study.py'})
    for name in snapshot:
        frozen = destination / 'source_at_run' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        clone_or_copy(ROOT / name, frozen)
    plan.update(runtime_snapshot=snapshot, runtime_entry=str(Path(__file__).resolve()))
    study.write_json(destination / 'protocol.json', plan)
    # Keep the already retained common archive at its original physical address.
    study.write_json(destination / 'initial_archive.json',
                     {'source_ref': registration['initial_archive_ref']})
    paid_traces = study.archived_paid_trace_references(knowledge.get('files', {}),
        destination / 'reference_archive', source_workspace=source / 'reference_archive')
    records, trials = [], []
    for job in registration['jobs']:
        if job['mode'] == 'retained_completed':
            records.append({**deepcopy(job), 'status': 'retained_completed',
                'costs': deepcopy(job['prior_costs']), 'continuation_cost': {
                    key: 0 for key in job['prior_costs']}})
            print(json.dumps({'case': job['case'], 'arm': job['arm'],
                'status': 'retained_completed', 'comparison_completed': None,
                'reused_score_count': 0, 'new_scored_count': 0,
                'continuation_compute_cost': 0}), flush=True)
            continue
        case = next(case for case in plan['cases'] if case['id'] == job['case'])
        previous = _value(job['training_ref'])
        expected_index = next(i for i in range(len(previous['method_results']) - 1, -1, -1)
            if previous['method_results'][i].get('proposal', {}).get('arguments', {}).get(
                'operator') == 'research_choice')
        expected = previous['method_results'][expected_index]['proposal']
        memory = deepcopy(knowledge)
        memory['finite_native_author']['question'] = deepcopy(
            plan['prepared_public_questions'][case['id']])
        work = deepcopy(organization)
        if job['arm'] == 'pr3':
            work['consolidation']['allocation']['compute'] = plan[
                'source_cohort_allocation']['local_consolidation_compute']
        limits = replace(study.InquiryLLMConfig(**plan['limits']),
                         compute_budget=job['compute_cap'])
        workspace = destination / case['id'] / job['arm']
        agent = study.RegisteredOrganizationAgent(None, workspace, arm=job['arm'],
            research_organization=work, knowledge=memory, config=limits,
            confirmation_seed=case['confirmation_seed'], source_task=False,
            paid_trace_references=paid_traces)
        laboratory = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
        agent.restore_research(previous, Path(job['training_ref']['file']).parent,
            laboratory, confirmation_seed=case['confirmation_seed'])
        before = deepcopy(agent.costs)
        preflight = _comparison_preflight(agent, laboratory, plan['author']['path'], expected)
        preflight['expected_proposal_ref'] = {'file': job['training_ref']['file'],
            'json_path': ['method_results', expected_index, 'proposal']}
        study.write_json(workspace / 'comparison_preflight.json', preflight)
        checkpoint = lambda data: study.write_json(workspace / 'active.json', data)

        def after_step(lab):
            agent._organization_state['finite_author']['completed_steps'] += 1
            agent._after_model_turn(lab)

        outcome = (execute_procedure(agent, laboratory, plan['author']['path'],
            max_steps=1, allow_confirmation=False, record_events=True,
            on_checkpoint=checkpoint, after_step=after_step)
            if preflight['exact_saved_proposal'] else {'status': 'comparison_not_executed'})
        result = agent.result(outcome['status'])
        result.update(prior_research_cost=before, resume_origin=job['training_ref']['file'],
            continuation_cost={key: value - before[key] for key, value in result['costs'].items()},
            comparison_preflight_ref={'file': str(workspace / 'comparison_preflight.json'),
                                      'json_path': []})
        checkpoint(result)
        study.write_json(workspace / 'training.json', result)
        study.write_json(workspace / 'archive.json', result['outgoing_knowledge'])
        feedback = (agent.method_results[-1]['feedback']
                    if len(agent.method_results) > len(previous['method_results']) else {})
        records.append({**deepcopy(job), 'status': result['status'], 'costs': result['costs'],
            'continuation_cost': result['continuation_cost'],
            'result_ref': {'file': str(workspace / 'training.json'), 'json_path': []},
            'comparison_completed': feedback.get('operator') == 'research_choice'
                and not feedback.get('error') and not feedback.get('truncated'),
            'reused_score_count': feedback.get('reused_score_count'),
            'new_scored_count': feedback.get('new_scored_count'),
            'retained_comparison_score_counts': feedback.get('retained_comparison_score_counts'),
            'new_observations': result['costs']['observation_cost'] - before['observation_cost']})
        print(json.dumps({key: records[-1][key] for key in ('case', 'arm', 'status',
            'comparison_completed', 'reused_score_count', 'new_scored_count')} |
            {'continuation_compute_cost': result['continuation_cost']['compute_cost']}), flush=True)
        trials.append({'case': case, 'arm': job['arm'], 'result': {
            key: result[key] for key in ('workspace', 'costs', 'research_organization')}})
    aggregation = study._aggregate_registered_sources(plan, destination, trials,
        organization, knowledge, paid_traces, previous_study=source)
    summary = {'status': 'completed', 'scope': registration['scope'], 'trials': records,
        'source_aggregation': aggregation, 'new_terminal_evaluations': 0,
        'new_observations': sum(record.get('new_observations', 0) for record in records),
        'continuation_compute_cost': sum(record['continuation_cost']['compute_cost']
            for record in records) + aggregation['new_compute_cost']}
    study.write_json(destination / 'summary.json', summary)
    return summary


def prepare_source_research(source_study, output, compute_increment=200000,
                            native_compute=100000):
    """Register a same-paid source continuation after the frozen comparison.

    All three groups receive the same increment and native invocation allowance.
    Each author may use its existing paid-bank continuation once; acquisitions,
    confirmations and successor studies are outside this stage.
    """
    from hegel_machine.cognition.finite_native_author import export_finite_native_author

    source, output = Path(source_study).resolve(), Path(output).resolve()
    if (output / 'research_registration.json').exists():
        raise ValueError('use a fresh source research preparation directory')
    if compute_increment <= 0 or native_compute <= 0:
        raise ValueError('register positive common compute allowances')
    plan, summary = _read(source / 'protocol.json'), _read(source / 'summary.json')
    initial = _read(source / 'initial_archive.json')['source_ref']
    plan['limits']['compute_budget'] += compute_increment
    cohort = plan['source_cohort_allocation']
    cohort['source_compute_limit'] += compute_increment * len(plan['cases'])
    for case in cohort['source_stage_compute']:
        cohort['source_stage_compute'][case] += compute_increment
    author_ref = {'file': str(output / 'registered_author.json'), 'json_path': ['source']}
    _write(output / 'registered_author.json', {'source': export_finite_native_author()})
    plan['author_source_ref'] = author_ref
    jobs = []
    for old in summary['trials']:
        retained = old['mode'] == 'retained_completed'
        jobs.append({'case': old['case'], 'arm': old['arm'],
            'mode': 'retained_completed' if retained else 'same_paid_source_research',
            'training_ref': deepcopy(old['training_ref'] if retained else old['result_ref']),
            'evaluation_ref': deepcopy(old['evaluation_ref']),
            'prior_costs': deepcopy(old['costs']), 'compute_cap': old['compute_cap'] + compute_increment})
    _write(output / 'research_plan.json', plan)
    registration = {'status': 'prepared_not_executed', 'source_study': str(source),
        'initial_archive_ref': initial, 'plan_ref': {
            'file': str(output / 'research_plan.json'), 'json_path': []},
        'source_summary_ref': {'file': str(source / 'summary.json'), 'json_path': []},
        'author_source_ref': author_ref, 'native_compute': native_compute,
        'compute_increment_per_original_job': compute_increment, 'jobs': jobs,
        'execution': {'same_paid_only': True, 'native_attempts_per_job': 1,
            'organization_hooks': True, 'new_observations': 0, 'allow_confirmation': False,
            'terminal_evaluations': 0, 'successor_execution': False},
        'scope': 'Continue original sources on their unchanged saved paid records, with equal '
                 'ABC caps and the same author repair. No fresh independent comparison.'}
    _write(output / 'research_registration.json', registration)
    return registration


def _source_research_steps(agent, laboratory, plan, checkpoint, *, native_attempt_limit=1):
    """Use the normal hooks/dispatcher, stopping before any new acquisition."""
    from hegel_machine.cognition.experiment_runtime import ExecutionTiming
    from hegel_machine.cognition.temporal_paired_method import execute_procedure

    author = plan['author']
    state = agent._organization_state['finite_author']
    start = len(agent.method_results)
    native_attempts, boundary = 0, None
    status = 'method_step_limit'
    timing = ExecutionTiming(('before_turn_hooks', 'prepare_state', 'author',
                           'procedure_including_after_and_checkpoint',
                           'after_turn_hooks_within_procedure', 'checkpoint_within_procedure'))

    def timed_checkpoint(data):
        with timing.phase('checkpoint_within_procedure'):
            return checkpoint(data)
    while state['completed_steps'] < author['max_steps']:
        exhausted = agent._procedure_budget_status()
        if exhausted:
            status = exhausted
            break
        # Source consolidation owns a separate theory question. Resume the
        # already registered public question, not that administrative owner.
        agent._active_question_id = agent.knowledge['finite_native_author']['question']['id']
        with timing.phase('before_turn_hooks'):
            agent._before_model_turn(laboratory)
        if agent._procedure_budget_status():
            status = agent._procedure_budget_status()
            break
        with timing.phase('prepare_state'):
            agent._prepare_program_state()
        path = author['path']
        source = (agent.workspace / path).read_text(encoding='utf-8')
        namespace = {'__name__': 'registered_source_author',
                     '__file__': str(agent.workspace / path)}
        started, previous_cwd = monotonic(), Path.cwd()
        try:
            os.chdir(agent.workspace)
            exec(compile(source, str(agent.workspace / path), 'exec'), namespace)
            proposal = namespace['select_actions'](laboratory.public_spec(),
                deepcopy(agent.observations), agent._program_memory(path, {'path': path}))
        finally:
            os.chdir(previous_cwd)
            author_seconds = monotonic() - started
            agent.costs['python_execution_seconds'] += author_seconds
            timing.phases['author'] += author_seconds
        prepared = {'path': path, 'proposal': proposal, 'program_execution': {
            'path': path, 'returncode': 0, 'timeout': False,
            'python_execution_seconds': author_seconds}}
        action, arguments = proposal.get('action'), proposal.get('arguments', {})
        if (action in {'query', 'finish'} or 'model' in proposal
                or not action and proposal.get('actions')):
            boundary = {'status': 'same_paid_boundary', 'proposed_action': proposal,
                'reason': 'New observations and confirmation are outside this registered stage.'}
            status = boundary['status']
            break
        operator = arguments.get('operator')
        if operator == 'native_condition_research' and native_attempts >= native_attempt_limit:
            boundary = {'status': 'registered_native_attempt_limit', 'proposed_action': proposal}
            status = boundary['status']
            break

        def after_step(lab):
            state['completed_steps'] += 1
            with timing.phase('after_turn_hooks_within_procedure'):
                agent._after_model_turn(lab)

        results_before_execution = len(agent.method_results)
        with timing.phase('procedure_including_after_and_checkpoint'):
            outcome = execute_procedure(agent, laboratory, path, max_steps=1,
                allow_confirmation=False, record_events=True,
                on_checkpoint=timed_checkpoint, after_step=after_step,
                _prepared_program_result=prepared)
        status = outcome['status']
        if len(agent.method_results) == results_before_execution:
            # The author may consume the final Python allowance before its
            # proposal can execute. Preserve that real cost and stop without
            # treating an older feedback record as this action's result.
            break
        if operator == 'native_condition_research':
            native_attempts += 1
        feedback = agent.method_results[-1]['feedback']
        if (native_attempts and feedback.get('operator') == 'research_choice'
                and not feedback.get('error') and not feedback.get('truncated')):
            status = 'same_paid_native_comparison_completed'
            break
        if status != 'method_step_limit':
            break
    return {'status': status, 'native_attempts': native_attempts,
            'new_method_results': len(agent.method_results) - start, 'boundary': boundary,
            'execution_timing': timing.result()}


def run_prepared_source_research(output):
    """Continue the registered sources; never dispatch targets or terminal tests."""
    from dataclasses import replace
    from hegel_machine.cognition.code_agent import clone_or_copy
    from scripts import run_anonymous_revision_inquiry as study

    output = Path(output).resolve()
    registration = _read(output / 'research_registration.json')
    source = Path(registration['source_study'])
    destination = output / 'source'
    if destination.exists():
        raise ValueError('retain the previous source continuation; use a fresh prepared directory')
    destination.mkdir(parents=True)
    plan = _value(registration['plan_ref'])
    initial = registration['initial_archive_ref']
    initial_source = Path(initial['file']).parent
    organization, knowledge = study._prepare_organization(
        plan, destination, _value(initial), initial_source)
    snapshot = sorted({str(path.relative_to(ROOT))
        for path in (ROOT / 'src/hegel_machine').rglob('*.py')} |
        {'scripts/run_anonymous_revision_inquiry.py', 'scripts/prepare_pr3_source_successor.py',
         'scripts/run_online_inquiry_llm.py', 'scripts/run_online_inquiry_study.py'})
    for name in snapshot:
        frozen = destination / 'source_at_run' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        clone_or_copy(ROOT / name, frozen)
    plan.update(runtime_snapshot=snapshot, runtime_entry=str(Path(__file__).resolve()))
    study.write_json(destination / 'protocol.json', plan)
    study.write_json(destination / 'initial_archive.json', {'source_ref': initial})
    paid_traces = study.archived_paid_trace_references(knowledge.get('files', {}),
        destination / 'reference_archive', source_workspace=initial_source / 'reference_archive')
    records, trials = [], []
    for job in registration['jobs']:
        if job['mode'] == 'retained_completed':
            result = {**deepcopy(job), 'status': 'retained_completed',
                'costs': deepcopy(job['prior_costs']),
                'continuation_cost': {key: 0 for key in job['prior_costs']}}
        else:
            case = next(c for c in plan['cases'] if c['id'] == job['case'])
            memory, work = deepcopy(knowledge), deepcopy(organization)
            memory['finite_native_author']['question'] = deepcopy(
                plan['prepared_public_questions'][case['id']])
            if job['arm'] == 'pr3':
                work['consolidation']['allocation']['compute'] = plan[
                    'source_cohort_allocation']['local_consolidation_compute']
            agent = study.RegisteredOrganizationAgent(None,
                destination / case['id'] / job['arm'], arm=job['arm'],
                research_organization=work, knowledge=memory,
                config=replace(study.InquiryLLMConfig(**plan['limits']),
                               compute_budget=job['compute_cap']),
                confirmation_seed=case['confirmation_seed'], source_task=False,
                paid_trace_references=paid_traces)
            lab = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
            previous = _value(job['training_ref'])
            agent.restore_research(previous, Path(job['training_ref']['file']).parent,
                                   lab, confirmation_seed=case['confirmation_seed'])
            before = deepcopy(agent.costs)
            agent.knowledge['finite_native_author']['max_compute'] = registration['native_compute']
            agent._save(plan['author']['path'], _value(registration['author_source_ref']))
            checkpoint = lambda data: study.write_json(agent.workspace / 'active.json', data)
            outcome = _source_research_steps(agent, lab, plan, checkpoint)
            data = agent.result(outcome['status'])
            delta = {key: value - before[key] for key, value in data['costs'].items()}
            data.update(prior_research_cost=before, resume_origin=job['training_ref']['file'],
                        continuation_cost=delta, same_paid_source_continuation=outcome)
            checkpoint(data)
            study.write_json(agent.workspace / 'training.json', data)
            study.write_json(agent.workspace / 'archive.json', data['outgoing_knowledge'])
            result = {**deepcopy(job), 'status': data['status'], 'costs': data['costs'],
                'continuation_cost': delta, **outcome,
                'result_ref': {'file': str(agent.workspace / 'training.json'), 'json_path': []}}
            trials.append({'case': case, 'arm': job['arm'], 'result': {
                key: data[key] for key in ('workspace', 'costs', 'research_organization')}})
        records.append(result)
        print(json.dumps({key: result.get(key) for key in
            ('case', 'arm', 'status', 'native_attempts', 'new_method_results', 'continuation_cost')}),
            flush=True)
    aggregation = study._aggregate_registered_sources(plan, destination, trials,
        organization, knowledge, paid_traces, previous_study=source)
    summary = {'status': 'completed', 'scope': registration['scope'], 'trials': records,
        'source_aggregation': aggregation, 'new_terminal_evaluations': 0,
        'new_observations': sum(j['continuation_cost']['observation_cost'] for j in records),
        'continuation_compute_cost': sum(j['continuation_cost']['compute_cost'] for j in records)
                                     + aggregation['new_compute_cost']}
    study.write_json(destination / 'summary.json', summary)
    return summary


def _saved_scene_feedback(value, reference):
    """Keep prior decision fields small; full failures/residuals retain their address."""
    # The finite author's reuse tests consume these decision fields. Large
    # historical route/forecast/binding payloads remain at their original ref.
    fields = ('operator', 'status', 'error', 'returncode', 'timeout', 'truncated',
        'research_question_id', 'question_meaning_version', 'paid_rows', 'evaluations',
        'evaluated_model', 'decision', 'recommendation', 'next_action', 'reused_score_count',
        'new_scored_count', 'retained_comparison_score_counts', 'compute_cost',
        'python_execution_seconds', 'new_compute_cost', 'native_condition_research',
        'native_theory_registration', 'returned_answer', 'record_indices', 'observations',
        'observation_cost', 'measurement_cost', 'path', 'tested_model', 'source_model',
        'program_action', 'executed_path', 'arguments', 'index')
    result = {key: deepcopy(value[key]) for key in fields if key in value}
    if isinstance(result.get('evaluations'), list):
        result['evaluations'] = [{key: deepcopy(item[key]) for key in
            ('path', 'returncode', 'mse', 'loss', 'loss_kind', 'truncated',
             'retention_decision', 'paid_score_reuse', 'evaluation_scope') if key in item}
            for item in result['evaluations']]
    # Frozen-program feedback repeats the same complete operator result inside
    # program_execution; the original addressed feedback retains that payload.
    result.pop('program_execution', None)
    result.pop('diagnosis', None)
    result['retained_full_feedback_ref'] = deepcopy(reference)
    return result


def _saved_scene_projection(source, training, state, seed):
    """Project the declared event19 scene, without importing final control state."""
    source = Path(source)
    selected = training['events'][19]['feedback']['evaluated_model']
    if selected['evaluated_rows'] != 97 or selected['current_paid_rows'] != 97:
        raise ValueError('event19 must retain the declared97-row scene')
    public = deepcopy(training['method_results'][9]['proposal']['research_question'])
    owner = next(q for q in state['questions'] if q['id'] == public['id'])
    banks = deepcopy(owner['native_condition_research'][:3])
    if (len(banks) != 3 or any(any(row >= 97 for row in bank['fit_evidence'])
                              for bank in banks)):
        raise ValueError('saved native bank exceeds the97-row cutoff')
    records = deepcopy(training['paid_observations'][:6])
    if sum(len(item['rows']) for item in records) != 97:
        raise ValueError('saved scene needs exactly the first six paid records')
    previous = deepcopy(seed)
    previous.update(model=None, fresh_confirmation=None, paid_observations=records,
        evaluated_model=deepcopy(selected), revision_proposals=[],
        experiment_comparisons=[], observable_models={}, conditional_models={},
        model_dependencies=[], research_input_bindings={}, research_candidates={},
        relation_research={}, research_mission={}, research_agenda={}, llm_inquiries={},
        inquiry_stack=[], program_contexts={})
    previous['events'] = []
    for i, event in enumerate(training['events'][:20]):
        copied = deepcopy(event)
        copied['feedback'] = _saved_scene_feedback(event.get('feedback', {}),
            {'file': str(source / 'training.json'), 'json_path': ['events', i, 'feedback']})
        previous['events'].append(copied)
    previous['method_results'] = []
    for i, entry in enumerate(training['method_results'][:10]):
        copied = deepcopy(entry)
        copied['feedback'] = _saved_scene_feedback(entry['feedback'],
            {'file': str(source / 'training.json'), 'json_path': ['method_results', i, 'feedback']})
        previous['method_results'].append(copied)
    # Rebuild control state from the original declaration. Later routing caches,
    # question history, failed searches and index updates are not copied back.
    public.update(meaning_version=1, opened_before_observation=72, evidence=[],
        status='open', experiment_path=None, native_condition_research=banks)
    answer = deepcopy(training['events'][18]['feedback']['answer_execution']['answer'])
    if answer.get('status') != 'answered' or answer.get('evidence') != [72]:
        raise ValueError('the original Readout answer must retain its row72 binding')
    public['semantic_answer'] = answer
    previous['online_questions'] = [deepcopy(seed['online_questions'][0]), public]
    # A named Readout needs the actual purchase binding, not just its saved
    # numeric answer. Keep only this meaning's executions within the paid prefix.
    for parent in previous['online_questions']:
        original_parent = next((item for item in state['questions']
                                if item['id'] == parent['id']), {})
        bindings = []
        for executed in original_parent.get('executed_pending_queries', []):
            origin = executed.get('source', {})
            indices = executed.get('record_indices', [])
            if (origin.get('kind') == 'question_selection' and origin.get('stage') != 2
                    and origin.get('question_id') == public['id']
                    and origin.get('meaning_version') == public['meaning_version']
                    and origin.get('experiment_id') and indices
                    and all(type(index) is int and 0 <= index < len(records) for index in indices)):
                bindings.append(deepcopy(executed))
        if bindings:
            parent['executed_pending_queries'] = bindings
    index = deepcopy(seed.get('theory_index', {}))
    registered = {entry['theory_id']: entry['version'] for bank in banks for entry in
                  bank.get('native_theory_registration', {}).get('registered', [])}
    for identifier, number in registered.items():
        latest = training['theory_index'][identifier]
        versions = [*latest.get('history', []), latest]
        found = [item for item in versions if item.get('version') == number]
        if len(found) != 1:
            raise ValueError('the saved native registration version is unavailable')
        original = found[0]
        src = original['source_ref']
        if src['round_index'] > 2:
            raise ValueError('native registration is not the saved prefix version')
        index[identifier] = {key: deepcopy(original[key]) for key in
            ('id', 'version', 'source_ref', 'role_signature', 'output_unit', 'context_ref',
             'program_source_ref') if key in original}
        index[identifier]['support'] = {'status': 'candidate', 'evidence': []}
    previous['theory_index'] = index
    operators = [_saved_scene_feedback(item, {'file': str(source / 'research_state.json'),
        'json_path': ['operator_results', i]}) for i, item in enumerate(state['operator_results'][:9])]
    if (len(operators) != 9 or operators[-1].get('operator') != 'research_choice'
            or operators[-1].get('paid_rows') != 97):
        raise ValueError('prefix operator boundary does not match event19')
    projected = {'questions': previous['online_questions'], 'operator_results': operators,
        'exported_candidate_paths': [], 'online_research': {'active_question': public['id']}}
    native = [candidate for bank in banks for candidate in bank.get('candidates', [])]
    models = list(dict.fromkeys(training['method_results'][0]['proposal']['arguments']['models']
                               + [candidate['model'] for candidate in native]))
    if len(models) != 1946 or len(native) != 14:
        raise ValueError('preserve the complete1932 old paths plus14 prefix native paths')
    return previous, projected, native, models


def _retain_scene_score_index(source, destination):
    """Reuse only immutable prior score scopes on at most the97 available rows."""
    original = Path(source) / '_paid_python_scores/_evaluated_model/assessments.json'
    if not original.is_file():
        return {'retained_scopes': 0, 'retained_assessments': 0}
    index = _read(original)
    scopes = [item for item in index['scopes'] if len(_value(item)['rows']) <= 97]
    eligible = {item['file'] for item in scopes}
    assessments = [item for item in index['assessments']
                   if _value(item)['scope_ref']['file'] in eligible]
    _write(Path(destination) / '_paid_python_scores/_evaluated_model/assessments.json',
           {'scopes': scopes, 'assessments': assessments, 'contexts': []})
    return {'retained_scopes': len(scopes), 'retained_assessments': len(assessments),
        'source_index_ref': {'file': str(original), 'json_path': []},
        'context_identity_still_checked_by_ordinary_scorer': True}


def _accept_saved_scene(agent, laboratory, manifest):
    """Check the real restored Readout and separate parent duty without recall."""
    from hegel_machine.cognition.question_continuation import (
        executed_plan_records, parent_revision_obligation, step_question,
    )

    before = deepcopy(agent.costs)
    receipt = {'status': 'failed', 'new_predictions': 0, 'new_fits': 0,
        'new_observations': 0, 'new_recalls': 0, 'source_readout_ref': manifest['old_readout_ref']}
    try:
        declaration = _value(manifest['old_readout_ref'])
        question = next(item for item in agent.questions if item['id'] == declaration['id'])
        identity_keys = ('id', 'meaning_version', 'kind', 'answer_type', 'context_ref',
                         'answer_resolution_ref', 'query_semantics')
        identity = {key: deepcopy(question[key]) for key in identity_keys if key in question}
        expected = {'status': 'answered', 'value': .21383203807282364, 'evidence': [72]}
        if (len(agent.rows) != 97 or len(agent.observations) != 6
                or question.get('meaning_version', 1) != 1
                or question.get('query_semantics') != declaration.get('query_semantics')
                or any(question.get('semantic_answer', {}).get(key) != value
                       for key, value in expected.items())):
            raise ValueError('saved Readout identity, answer or97-row boundary is not retained')
        record_selector = declaration['query_semantics']['query_expression']['record_index']
        bindings = executed_plan_records(agent, question)
        if (record_selector != {'experiment_id': 'public_positive_half_reset', 'action_index': 0}
                or bindings.get(record_selector['experiment_id']) != [3]):
            raise ValueError('saved named experiment must bind to actual paid record3')
        # This is the ordinary pre-turn metadata binding; no route/recall occurs.
        agent._bind_revision_target(question)
        if question.get('theory_target', {}).get('parent_theory_id') != manifest['parent_theory_id']:
            raise ValueError('revision target is not the actually retained event19 parent')
        answer_execution = step_question(agent, question, laboratory)
        obligation = parent_revision_obligation(agent, question, answer_execution)
        source = {'question_id': question['id'], 'meaning_version': question.get('meaning_version', 1)}
        if ({key: question[key] for key in identity} != identity
                or any(question.get('semantic_answer', {}).get(key) != value
                       for key, value in expected.items())
                or any(answer_execution.get('answer', {}).get(key) != value
                       for key, value in expected.items())):
            raise ValueError('paid Readout execution changed its identity or correct answer')
        if (obligation is None or obligation['id'] == question['id']
                or obligation.get('kind') != 'program_revision'
                or obligation.get('native_research_ref') != source
                or obligation.get('origin') != {
                    'kind': 'unfinished_parent_revision', 'source_question_ref': source}
                or obligation.get('theory_target', {}).get('parent_theory_id') != manifest['parent_theory_id']
                or question.get('parent_revision_question_ref', {}).get('question_id') != obligation['id']):
            raise ValueError('independent unfinished parent revision obligation is absent or misbound')
        delta = {key: agent.costs[key] - value for key, value in before.items()}
        if any(value for key, value in delta.items() if key != 'compute_cost'):
            raise ValueError('scene acceptance must only charge paid-evidence readout compute')
        receipt.update(status='accepted', question_identity=identity,
            semantic_answer=deepcopy(question['semantic_answer']), named_record_bindings=bindings,
            answer_execution=answer_execution,
            parent_obligation={key: deepcopy(obligation[key]) for key in
                ('id', 'meaning_version', 'kind', 'answer_type', 'origin', 'native_research_ref', 'subject_refs')},
            parent_theory_id=manifest['parent_theory_id'], new_costs=delta,
            no_scientific_loop_started=True)
    except Exception as error:
        receipt.update(error=f'{type(error).__name__}: {error}',
            new_costs={key: agent.costs[key] - value for key, value in before.items()})
        _write(agent.workspace / 'scene_acceptance.json', receipt)
        agent._prepare_program_state()
        _write(agent.workspace / 'scene_acceptance_checkpoint.json', agent.result('scene_acceptance_failed'))
        raise
    _write(agent.workspace / 'scene_acceptance.json', receipt)
    agent._prepare_program_state()
    _write(agent.workspace / 'scene_acceptance_checkpoint.json', agent.result('scene_accepted'))
    return receipt


def prepare_saved_scene_interventions(preparation, output, *, adopted_candidate_id,
        adoption_refs, library_ref, index_ref, redundancy_classes_ref, source_cost_ref):
    """Register actual-use removals without restoring or executing the scene."""
    from hegel_machine.cognition.theory_consolidation import _read_index, _read_library

    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('retain the previous intervention registration; use a fresh output')
    manifest = _read(preparation / 'scene_manifest.json')
    baseline = _read_library(manifest['library_views']['B_old8'], ROOT)
    library = _read_library(library_ref, ROOT)
    expected = manifest['library_views']['old_ids'] + manifest['library_views']['new_ids']
    if (list(library.get('candidates', {})) != expected
            or set(baseline.get('candidates', {})) != set(manifest['library_views']['old_ids'])
            or adopted_candidate_id not in manifest['library_views']['new_ids'] or not adoption_refs):
        raise ValueError('register an actually adopted new object in the complete original18 view')
    groups = _value(redundancy_classes_ref)
    selected = [(position, group) for position, group in enumerate(groups)
        if adopted_candidate_id in [*group.get('original_member_ids', []),
                                    *group.get('additional_same_contract_member_ids', [])]]
    if len(selected) != 1:
        raise ValueError('the adopted object must belong to exactly one retained redundancy class')
    position, group = selected[0]
    aliases = [*group.get('original_member_ids', []), *group.get('additional_same_contract_member_ids', [])]
    if not set(aliases) <= set(manifest['library_views']['new_ids']):
        raise ValueError('the retained class may remove only the registered new objects')
    source = Path(manifest['source_training_ref']['file']).parent
    plan = _read(source.parent.parent / 'protocol.json')
    plan['source_cost_ref'] = deepcopy(source_cost_ref)
    plan['author'] = {**plan['author'], 'max_steps': 8, 'confirm': False}
    indexed = _read_index(index_ref, ROOT)
    keep = output / 'interventions' / 'keep' / 'theory_library.json'
    _write(keep, {'source_ref': deepcopy(library_ref)})
    branches = {'keep': {'organization_arm': 'pr3',
        'library_ref': {'file': str(keep), 'json_path': []}, 'index_ref': deepcopy(index_ref),
        'expected_candidate_ids': expected, 'removed_candidate_ids': []}}
    class_ref = {**deepcopy(redundancy_classes_ref),
                 'json_path': [*redundancy_classes_ref.get('json_path', []), position]}
    for label, kind, identifiers, retained_class in (
            ('individual_remove', 'individual_object', [adopted_candidate_id], None),
            ('class_remove', 'new_redundancy_class', aliases, class_ref)):
        frozen, _, _ = _freeze_removal(plan, library_ref, library, baseline, index_ref,
            indexed, identifiers, output, kind, class_ref=retained_class)
        removal_plan = _value(frozen['plan_ref'])
        branches[label] = {'organization_arm': 'pr3',
            'library_ref': frozen['frozen_remove_library_ref'],
            'index_ref': deepcopy(removal_plan['target_intervention'].get('index_view_ref') or index_ref),
            'manifest_ref': frozen['manifest_ref'], 'removed_candidate_ids': sorted(identifiers),
            'expected_candidate_ids': [identifier for identifier in expected if identifier not in identifiers]}
    registration = {'status': 'registered_not_executed', 'scope': 'saved97_development_actual_object_removal',
        'preparation': str(preparation), 'run_output': str(output / 'run'),
        'source_manifest_ref': {'file': str(preparation / 'scene_manifest.json'), 'json_path': []},
        'adopted_candidate_id': adopted_candidate_id, 'adoption_refs': deepcopy(adoption_refs),
        'retained_class_ref': class_ref, 'source_cost_ref': deepcopy(source_cost_ref),
        'common_initial_index_ref': deepcopy(index_ref), 'branches': branches,
        'additional_compute_per_arm': 1500000, 'additional_python_seconds_per_arm': 1800,
        'native_attempt_limit_per_arm': 8, 'maximum_physical_cores': 8,
        'paid_rows': 97, 'original_pool_count': 1932, 'native_prefix_pool_count': 14,
        'post97_prediction_contract': deepcopy(manifest.get('post97_prediction_contract')),
        'keep_reruns_on_common_current_runtime': True, 'old_keep_is_new_counterfactual': False,
        'new_observations': 0, 'new_model_calls': 0, 'source_costs_recharged': False}
    _write(output / 'intervention_registration.json', registration)
    return registration


def run_saved_scene_interventions(registration_ref, *, workers=8):
    """Execute the registered keep/individual/class scenes through the same loop."""
    reference = ({'file': str(Path(registration_ref).resolve()), 'json_path': []}
                 if isinstance(registration_ref, (str, Path)) else deepcopy(registration_ref))
    registration = _value(reference)
    if (registration['status'] != 'registered_not_executed'
            or tuple(registration['branches']) != ('keep', 'individual_remove', 'class_remove')
            or not 1 <= workers <= registration['maximum_physical_cores']):
        raise ValueError('execute only the registered three branches within the eight-core cap')
    accepted = accept_prepared_saved_scene(registration['preparation'], registration['run_output'],
        branch_views=registration['branches'])
    accepted['intervention_registration_ref'] = reference
    accepted['source_cost_ref'] = deepcopy(registration['source_cost_ref'])
    return run_prepared_saved_scene(registration['preparation'], registration['run_output'],
        accepted_scene=accepted, workers=workers)


def accept_prepared_saved_scene(preparation, output=None, *, branch_views=None):
    """Restore and accept registered scenes, retaining live agents for execution."""
    from dataclasses import replace
    from hegel_machine.cognition.checkpoint_storage import clone_or_copy
    from hegel_machine.cognition.code_agent import _closed_temporal_export, python_model_context
    from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.finite_native_author import ordinary_completion_reserve
    from scripts import run_anonymous_revision_inquiry as study

    preparation = Path(preparation).resolve()
    manifest = _read(preparation / 'scene_manifest.json')
    source = Path(manifest['source_training_ref']['file']).parent
    source_study = source.parent.parent
    output = Path(output or preparation / 'diagnostic').resolve()
    if output.exists():
        raise ValueError('retain the previous diagnostic; use a fresh output directory')
    output.mkdir(parents=True)
    plan = _read(source_study / 'protocol.json')
    original = _read(source / 'training.json')
    state = _read(source / 'research_state.json')
    carry = deepcopy(original['costs'])
    limits = replace(study.InquiryLLMConfig(**plan['limits']),
        compute_budget=carry['compute_cost'] + 1500000,
        execution_seconds=carry['python_execution_seconds'] + 1800)
    organization, knowledge = study._prepare_organization(plan, output,
        _read(source_study / 'initial_archive.json'))
    paid_traces = study.archived_paid_trace_references(knowledge.get('files', {}),
        output / 'reference_archive', source_workspace=source_study / 'reference_archive')
    case = next(item for item in plan['cases'] if item['id'] == 'case_001')
    plan['author'] = {**plan['author'], 'max_steps': 8, 'confirm': False}
    def restore_arm(arm):
        workspace = output / arm
        selected = (branch_views or {}).get(arm, {})
        local_organization, local_knowledge = deepcopy(organization), deepcopy(knowledge)
        if selected:
            local_organization['pr3_library_path'] = selected['library_ref']['file']
            if selected.get('index_ref'):
                local_knowledge['theory_index'] = deepcopy(selected['index_ref'])
        agent = study.RegisteredOrganizationAgent(None, workspace,
            arm=selected.get('organization_arm', arm),
            research_organization=local_organization, knowledge=local_knowledge,
            config=limits, confirmation_seed=case['confirmation_seed'], source_task=False,
            paid_trace_references=paid_traces)
        # The seed is initialized from the frozen pre-target archive, never the
        # final target's writable model parameters or final index/question state.
        agent._prepare_program_state()
        seed = agent.result('saved_scene_initialization')
        previous, projected, native, models = _saved_scene_projection(source, original, state, seed)
        previous['costs'] = deepcopy(carry)
        previous['research_organization'] = deepcopy(agent._organization_state)
        previous['research_organization']['finite_author'] = {
            'contract': deepcopy(plan['author']), 'completed_steps': 0}
        projection = output / 'projection' / arm
        projection.mkdir(parents=True)
        for candidate in native:
            program = Program.from_dict(candidate['program'])
            directory, model = python_model_context(source, candidate['model'])
            if (not _closed_temporal_export(directory, model)
                    or (directory / model).read_text() != export_temporal_program(program,
                        feature_names=tuple(sorted(program.required_features())))):
                raise ValueError('prefix native model does not match its saved literal Program')
            destination = projection / candidate['model']
            destination.parent.mkdir(parents=True, exist_ok=True)
            clone_or_copy(directory / model, destination)
        retained = projection / previous['evaluated_model']['model']
        retained.parent.mkdir(parents=True, exist_ok=True)
        clone_or_copy(projection / previous['evaluated_model']['source_model'], retained)
        _write(projection / 'research_state.json', projected)
        lab = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
        restored = agent.restore_research(previous, projection, lab,
            confirmation_seed=case['confirmation_seed'])
        expected_parent = previous['evaluated_model']['source_model']
        if (len(agent.rows) != 97 or len(agent.observations) != 6
                or agent.evaluated_model['source_model'] != expected_parent
                or any(src.get('source_ref', {}).get('round_index', 0) > 2
                       for key, src in agent.theory_index.items()
                       if key.startswith('native:native_public_response@meaning1:')
                       and src.get('context_ref') == agent.questions[1].get('context_ref'))):
            raise ValueError('restored scene contains future state or a different parent')
        if models[:1932] != original['method_results'][0]['proposal']['arguments']['models']:
            raise ValueError('the complete original1932 predictor order must be retained')
        missing = [model for model in models if not (workspace / model).is_file()]
        if missing:
            _write(workspace / 'missing_predictors.json', missing)
            raise ValueError(f'the strong saved pool has{len(missing)} unavailable predictor paths')
        cache = _retain_scene_score_index(source, workspace)
        agent.knowledge['finite_native_author']['question'] = deepcopy(
            original['method_results'][9]['proposal']['research_question'])
        reserve = ordinary_completion_reserve(agent, agent.questions[1])
        validation = {'status': 'restored_without_scientific_execution', **restored,
            'paid_rows': len(agent.rows), 'native_paths': len(native), 'complete_paths': len(models),
            'original1932_order_retained': True, 'all1946_predictor_paths_exist': True,
            'ordinary_completion_reserve': reserve, 'retained_score_index': cache,
            'scene_is_explicit_control_projection_not_literal_checkpoint': True,
            'discarded_final_fields': ['routing/question caches', 'native rounds3/4',
                'future operator_results/index versions/exports', 'future48labels', 'final model parameters'],
            'shared_sunk_costs': carry, 'new_predictions': 0, 'new_fits': 0}
        _write(workspace / 'materialization_validation.json', validation)
        validation['acceptance'] = _accept_saved_scene(agent, lab, manifest)
        return {'agent': agent, 'lab': lab, 'previous': previous, 'validation': validation,
                'starting_costs': deepcopy(carry)}
    arms = {}
    try:
        for arm in (branch_views or ('fixed_pr2', 'pr3')):
            arms[arm] = restore_arm(arm)
    except Exception as error:
        _write(output / 'scene_acceptance.json', {'status': 'failed',
            'accepted_arms': list(arms), 'error': f'{type(error).__name__}: {error}',
            'scientific_loop_started': False})
        raise
    acceptance = {'status': ('all_registered_branches_accepted_before_scientific_execution'
                             if branch_views else 'both_arms_accepted_before_scientific_execution'),
        'arms': {arm: value['validation']['acceptance'] for arm, value in arms.items()},
        'new_costs': {key: sum(value['validation']['acceptance']['new_costs'][key]
                              for value in arms.values()) for key in carry},
        'scientific_loop_started': False, 'runtime_frozen': False}
    _write(output / 'scene_acceptance.json', acceptance)
    return {'preparation': preparation, 'output': output, 'manifest': manifest,
        'source': source, 'plan': plan, 'case': case, 'carry': carry, 'limits': limits,
        'arms': arms, 'acceptance': acceptance, 'branch_views': deepcopy(branch_views)}


def _freeze_saved_scene_libraries(accepted_scene):
    """Freeze each complete registered view actually used by accepted agents."""
    from hegel_machine.cognition.theory_consolidation import (
        _read_library, install_theory_library, _save as save_theory_library,
    )

    output, manifest = accepted_scene['output'], accepted_scene['manifest']
    frozen = {}
    for arm, retained in accepted_scene['arms'].items():
        agent = retained['agent']
        selected = agent._organization_state['automatic_library_path']
        reference = selected if isinstance(selected, dict) else {'file': selected, 'json_path': []}
        library = _read_library(reference, agent.workspace)
        selected_view = (accepted_scene.get('branch_views') or {}).get(arm)
        expected = (selected_view['expected_candidate_ids'] if selected_view else
            manifest['library_views']['old_ids'] + (
                manifest['library_views']['new_ids'] if arm == 'pr3' else []))
        if list(library.get('candidates', {})) != expected:
            raise ValueError('freeze must retain the exact registered saved-scene library view')
        path = output / 'frozen_libraries' / f'{arm}.json'
        _write(path, library)
        frozen_ref = {'file': str(path), 'json_path': []}
        agent.theory_consolidation = install_theory_library(agent, frozen_ref)
        save_theory_library(agent)
        agent._organization_state['automatic_library_path'] = str(path)
        for question in agent.questions:
            if question.get('frozen_library_path') == selected:
                question['frozen_library_path'] = str(path)
        frozen[arm] = {'library_ref': frozen_ref, 'candidate_count': len(expected),
            'candidate_versions': [{'id': key, 'version': item.get('version', 1)}
                                   for key, item in library['candidates'].items()],
            'source_library_ref': deepcopy(reference), 'expanded_not_thin_reference': True}
        agent._prepare_program_state()
    _write(output / 'frozen_libraries.json', frozen)
    return frozen


def prepare_retained_scene_continuation(source_training_ref, output, *,
        compute_increment=800000, python_seconds=1350., additional_steps=2):
    """Register two future-library views of one actual retained research state.

    Unlike the event19 projection, this keeps the already consumed candidates,
    parent, counterexamples and unfinished historical work in both branches.
    Preparation reads the small completed summary, not the training archive.
    """
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('retain the previous continuation; use a fresh registration')
    if (type(compute_increment) is not int or compute_increment <= 0
            or python_seconds <= 0 or type(additional_steps) is not int or additional_steps <= 0):
        raise ValueError('register positive common continuation allowances')
    reference = ({'file': str(Path(source_training_ref).resolve()), 'json_path': []}
                 if isinstance(source_training_ref, (str, Path)) else deepcopy(source_training_ref))
    source = Path(reference['file']).resolve().parent
    prior = _read(source.parent / 'protocol.json')
    summary = _read(source.parent / 'summary.json')
    row = next(item for item in summary['trials']
        if Path((item.get('result_ref') or item.get('training_ref'))['file']).resolve()
        == Path(reference['file']).resolve())
    if summary['status'] != 'completed' or row['arm'] != 'pr3':
        raise ValueError('continuation requires the completed retained C research state')
    manifest_ref = prior['source_manifest_ref']
    manifest = _value(manifest_ref)
    origin = Path(manifest['source_training_ref']['file']).resolve().parent
    registration = {'status': 'registered_not_executed',
        'scope': 'post_consumption_same_retained_C_future_old8_full18_development_control',
        'source_training_ref': reference,
        'source_summary_ref': {'file': str(source.parent / 'summary.json'), 'json_path': []},
        'source_manifest_ref': deepcopy(manifest_ref),
        'source_plan_ref': {'file': str(origin.parent.parent / 'protocol.json'), 'json_path': []},
        'source_initial_archive_ref': {
            'file': str(origin.parent.parent / 'initial_archive.json'), 'json_path': []},
        'run_output': str(output / 'run'),
        'starting_costs': deepcopy(row['costs']),
        'retained_training_costs': deepcopy(row.get('training_only_costs', row['costs'])),
        'source_original_limits': deepcopy(prior.get('limits', {})),
        'source_original_remaining_compute': (
            prior['limits']['compute_budget'] - row['costs']['compute_cost']
            if 'limits' in prior else None),
        'prior_prediction_receipt_ref': {
            'file': str(source.parent / 'post97_predictions.json'), 'json_path': []},
        'additional_compute_per_arm': compute_increment,
        'additional_python_seconds_per_arm': python_seconds,
        'additional_method_steps_per_arm': additional_steps,
        'compute_budget_rationale': 'Common allowance covers two complete retained-pool comparisons '
            'and two unchanged routing allocations without assuming score reuse; actual reused work '
            'remains free and unused allowance is not charged.',
        'maximum_physical_cores': 8, 'new_observations': 0, 'new_model_calls': 0,
        'common_scheduler_repair_is_PR3_effect': False,
        'prior_consumed_candidates_and_scores_retained_in_both_arms': True,
        'comparison_is_original_B_C_replay': False,
        'post97_prediction_contract': deepcopy(manifest.get('post97_prediction_contract')),
        'registered_arms': ['fixed_pr2', 'pr3'],
        'library_views': deepcopy(manifest['library_views'])}
    _write(output / 'retained_scene_registration.json', registration)
    return registration


def accept_prepared_retained_scene(registration, *, output=None):
    """Restore the literal completed C state twice, without fitting or scoring."""
    from dataclasses import replace
    from hegel_machine.cognition.method_effect_query import retained_research_progress
    from hegel_machine.cognition.research_episodes import SEMANTIC_FIELDS, _question_versions
    from hegel_machine.cognition.theory_consolidation import (
        LIBRARY_PATH, install_theory_library, _save as save_library)
    from scripts import run_anonymous_revision_inquiry as study

    registration_ref = ({'file': str(Path(registration).resolve()), 'json_path': []}
                        if isinstance(registration, (str, Path)) else deepcopy(registration))
    registered = _value(registration_ref)
    output = Path(output or registered['run_output']).resolve()
    if output.exists():
        raise ValueError('restore into a fresh workspace and retain the original research')
    source = Path(registered['source_training_ref']['file']).resolve().parent
    original = _value(registered['source_training_ref'])
    if original['costs'] != registered['retained_training_costs']:
        raise ValueError('registered training account differs from the actual saved state')
    carry = deepcopy(registered['starting_costs'])
    # The completed frozen suffix prediction was charged after training.json.
    # Carry that already incurred fee too; do not add it again per continuation.
    original['costs'] = deepcopy(carry)
    plan = _value(registered['source_plan_ref'])
    original_steps = original['research_organization']['finite_author']['completed_steps']
    plan['author'] = {**plan['author'], 'confirm': False,
        'max_steps': original_steps + registered['additional_method_steps_per_arm']}
    case = next(item for item in plan['cases'] if item['kind'] == 'delay')
    limits = replace(study.InquiryLLMConfig(**plan['limits']),
        compute_budget=carry['compute_cost'] + registered['additional_compute_per_arm'],
        execution_seconds=carry['python_execution_seconds'] + registered['additional_python_seconds_per_arm'])
    output.mkdir(parents=True)
    organization, knowledge = study._prepare_organization(
        plan, output, _value(registered['source_initial_archive_ref']))
    # The initial common archive intentionally has no task-specific question.
    # Reuse the same published per-case declaration as _prepared_callbacks.
    declared = plan['prepared_public_questions'][case['id']]
    current_public = next(question for question in original['online_questions']
                          if question['id'] == declared['id'])
    declaration_fields = ('id', 'parent_id', 'depends_on', 'meaning_version',
        'mission_id', 'subject_refs', 'theory_target', 'use_sites',
        'native_research_ref', 'resolution_experiments', *SEMANTIC_FIELDS)
    # The author declares the current public contract. Executed answers,
    # routing caches and native banks remain only in the restored live state.
    knowledge['finite_native_author']['question'] = {
        field: deepcopy(current_public[field] if field in current_public else declared[field])
        for field in declaration_fields if field in current_public or field in declared}
    source_study = Path(registered['source_plan_ref']['file']).parent
    paid = study.archived_paid_trace_references(knowledge.get('files', {}),
        output / 'reference_archive', source_workspace=source_study / 'reference_archive')
    source_library_ref = {'file': str(source / LIBRARY_PATH), 'json_path': []}
    expected_all = [*registered['library_views']['old_ids'], *registered['library_views']['new_ids']]
    frozen, arms = {}, {}
    for arm in registered['registered_arms']:
        expected = (registered['library_views']['old_ids'] if arm == 'fixed_pr2' else expected_all)
        path = output / 'frozen_libraries' / f'{arm}.json'
        _write(path, {'source_ref': source_library_ref,
            'removed_candidates': [identifier for identifier in expected_all if identifier not in expected]})
        local_organization = deepcopy(organization)
        local_organization['pr3_library_path'] = str(path)
        # Both branches use the same C hook and differ only in future recall.
        agent = study.RegisteredOrganizationAgent(None, output / arm, arm='pr3',
            research_organization=local_organization, knowledge=deepcopy(knowledge),
            config=limits, confirmation_seed=case['confirmation_seed'], source_task=False,
            paid_trace_references=paid)
        lab = study.AnonymousHistoryLab(kind=case['kind'], seed=case['seed'])
        restored = agent.restore_research(original, source, lab,
            confirmation_seed=case['confirmation_seed'])
        if len(agent.rows) != 97 or len(agent.observations) != 6:
            raise ValueError('retained continuation must use the same six-record97-row material')
        agent.theory_consolidation = install_theory_library(agent, {'file': str(path), 'json_path': []})
        if list(agent.theory_consolidation.get('candidates', {})) != list(expected):
            raise ValueError('the retained source library does not match its registered8/18 views')
        agent._organization_state['automatic_library_path'] = str(path)
        agent._organization_state['finite_author']['contract'] = deepcopy(plan['author'])
        agent.knowledge['finite_native_author']['question']['frozen_library_path'] = str(path)
        for question, _reference in _question_versions(agent).values():
            if ('frozen_library_path' in question or 'frozen_recall_execution' in question
                    or question.get('theory_target', {}).get('revision_fitting')):
                question['frozen_library_path'] = str(path)
        save_library(agent)
        cache = _retain_scene_score_index(source, agent.workspace)
        agent._prepare_program_state()
        event_start, method_start = len(agent.events), len(agent.method_results)
        progress = retained_research_progress(agent, event_start=event_start, method_start=method_start)
        _write(agent.workspace / 'research_progress_before.json', progress)
        validation = {'status': 'literal_retained_state_restored_without_scientific_execution',
            **restored, 'new_fits': 0, 'new_predictions': 0, 'new_observations': 0,
            'current_parent_retained': (
                agent.evaluated_model['source_model'] == original['evaluated_model']['source_model']
                and agent.evaluated_model['current_mse'] == original['evaluated_model']['mse']),
            'initial_method_steps': original_steps, 'retained_score_index': cache,
            'active_candidate_ids': list(expected),
            'complete_predictor_pool_count': progress['full_pool_count'],
            'score_reuse_checked_by': 'ordinary_scorer_at_actual_comparison',
            'progress_ref': {'file': str(agent.workspace / 'research_progress_before.json'), 'json_path': []},
            'acceptance': {'status': 'passed', 'new_costs': {key: 0 for key in carry}}}
        if not validation['current_parent_retained']:
            raise ValueError('literal restoration must retain the actual C task winner')
        _write(agent.workspace / 'materialization_validation.json', validation)
        _write(agent.workspace / 'scene_acceptance.json', validation['acceptance'])
        frozen[arm] = {'library_ref': {'file': str(path), 'json_path': []},
            'candidate_count': len(expected), 'candidate_versions': [
                {'id': identifier, 'version': candidate.get('version', 1)}
                for identifier, candidate in agent.theory_consolidation['candidates'].items()],
            'source_library_ref': source_library_ref, 'expanded_not_thin_reference': False,
            'immutable_retained_source_reused': True}
        arms[arm] = {'agent': agent, 'lab': lab, 'previous': original,
            'validation': validation, 'starting_costs': deepcopy(carry),
            'event_start': event_start, 'method_start': method_start}
    manifest = _value(registered['source_manifest_ref'])
    acceptance = {'status': 'both_arms_accepted_before_scientific_execution',
        'scientific_loop_started': False, 'literal_retained_state': True,
        'initial_method_steps': original_steps, 'additional_method_steps': registered['additional_method_steps_per_arm']}
    _write(output / 'scene_acceptance.json', acceptance)
    _write(output / 'frozen_libraries.json', frozen)
    return {'preparation': Path(registration_ref['file']).parent, 'output': output,
        'manifest': manifest, 'source': source, 'plan': plan, 'case': case,
        'carry': carry, 'limits': limits, 'arms': arms, 'acceptance': acceptance,
        'retained_scene_registration_ref': registration_ref,
        'retained_scene_registration': registered, 'retained_frozen_libraries': frozen}


def run_prepared_retained_scene(registration, *, workers=8, accepted_scene=None):
    """Execute the registered bounded continuation through the existing loop."""
    accepted = (accept_prepared_retained_scene(registration) if accepted_scene is None else accepted_scene)
    return run_prepared_saved_scene(None, accepted_scene=accepted, workers=workers)


def _saved_scene_arm_result(arm, data, workspace, new_ids, acceptance_cost, candidate_paths):
    """Summarize a persisted branch without executing its author or predictor."""
    from hegel_machine.cognition.method_effect_query import event_experiment_selection

    workspace = Path(workspace)
    before, delta = data['prior_research_cost'], data['continuation_cost']
    if delta['observation_cost'] or data.get('fresh_confirmation'):
        raise ValueError('same-paid development diagnostic acquired new observations')
    outcome = data['same_paid_source_continuation']
    events = data['events'][data['saved_scene_cutoff']['event'] + 1:]
    selected = [selection['selected_experiment'] for event in events
                if (selection := event_experiment_selection(event)).get('selected_experiment')]
    usage = {identifier: {'attempts': 0, 'generated_theory_ids': []} for identifier in new_ids}
    for event in events:
        for item in ((event.get('feedback') or {}).get('frozen_recall') or {}).get('work', []):
            if item['candidate_id'] in usage:
                usage[item['candidate_id']]['attempts'] += 1
                usage[item['candidate_id']]['generated_theory_ids'].extend(item.get('theory_ids', []))
    return {'case': data['saved_scene_cutoff'].get('case', 'saved_delay_event19'),
        'arm': arm, 'status': data['status'],
        'acceptance_cost': acceptance_cost,
        'acceptance_ref': {'file': str(workspace / 'scene_acceptance.json'), 'json_path': []},
        'starting_costs': before, 'costs': data['costs'], 'additional_costs': delta,
        'continuation_cost': delta, 'selected_parent': data.get('evaluated_model'),
        'candidate_paths': list(candidate_paths), 'selected_experiments': selected,
        'stop_reason': outcome['status'], 'new_object_usage': usage,
        'materialization_validation_ref': {'file': str(workspace / 'materialization_validation.json'), 'json_path': []},
        'result_ref': {'file': str(workspace / 'training.json'), 'json_path': []}, **outcome}


def _saved_scene_summary(results, carry):
    return {'status': 'completed', 'scientific_status': 'development_only_no_confirmation_or_terminal',
        'scope': 'two_registered_saved_scene_jobs_finished_not_original_independent_D3', 'trials': results,
        'shared_original_sunk_costs_once': carry,
        'additional_costs': {key: sum(row['continuation_cost'][key] for row in results) for key in carry},
        'new_terminal_evaluations': 0, 'new_confirmation_rows': 0,
        'comparison': {'candidate_change': None, 'parent_change': None,
            'experiment_change': None, 'post97_prediction_change': None,
            'reason': 'read_saved_result_and_common_inputs_before_interpreting_changes'},
        'post97_prediction_diagnostic': 'not_executed_candidates_require_separate_freeze'}


def _update_saved_scene_effects(result, effects, output):
    results = result['trials']
    if effects.get('cases'):
        mechanism = effects['cases'][0].get('mechanism', {})
        executable = mechanism.get('selected_parent_executable_comparison') or {}
        result['comparison'].update(
            candidate_change=mechanism.get('candidate_programs_changed'),
            parent_change=(executable.get('changed') if executable.get('status') == 'recorded'
                           else mechanism.get('parent_program_changed')),
            parent_reference_change=mechanism.get('parent_model_ids_changed'),
            parent_typed_program_change=mechanism.get('parent_program_changed'),
            experiment_change=mechanism.get('selected_experiment_actions_changed'),
            selected_parent_executable_comparison=mechanism.get('selected_parent_executable_comparison'),
            stored_forecast_comparison=mechanism.get('stored_forecast_comparison'),
            selected_parent_forecast_comparison=mechanism.get('selected_parent_forecast_comparison'),
            readout_ref={'file': str(output / 'research_effects.json'), 'json_path': ['cases', 0]})
        result['comparison']['reason'] = 'existing_read_only_effect_extractor_no_new_predictions_or_scores'
        for trial, objects in zip(results, mechanism.get('objects', [])):
            for identifier, usage in trial['new_object_usage'].items():
                retained = objects.get(identifier)
                usage.update(
                    status=retained.get('usage_status') if retained else 'not_in_active_library_view',
                    selection_count=len(retained.get('selection_refs', [])) if retained else 0,
                    adopted_parent_count=len(retained.get('adopted_parent_refs', [])) if retained else 0)


def recover_saved_scene_reports(output):
    """Read finished branch files after a reporting failure; write or execute nothing.

    The original failed run status remains evidence. The returned summary and
    effects can be retained separately; unexecuted post97 predictions stay unmeasured.
    """
    from hegel_machine.cognition.method_effect_query import registered_successor_effect

    output = Path(output).resolve()
    protocol = _read(output / 'protocol.json')
    manifest = _value(protocol['source_manifest_ref'])
    results = []
    for arm in ('fixed_pr2', 'pr3'):
        workspace = output / arm
        data = _read(workspace / 'training.json')
        acceptance = _read(workspace / 'scene_acceptance.json')
        # Read the large training and state documents sequentially. Retain only
        # their small result metadata, never two expanded experiment histories.
        row = _saved_scene_arm_result(arm, data, workspace,
            manifest['library_views']['new_ids'], acceptance['new_costs'], [])
        del data
        state = _read(workspace / 'research_state.json')
        row['candidate_paths'] = list(state['exported_candidate_paths'])
        del state
        results.append(row)
    summary = _saved_scene_summary(results, protocol['original_sunk_costs_once'])
    effects = registered_successor_effect(output, saved_summaries=[summary, summary])
    _update_saved_scene_effects(summary, effects, output)
    summary['comparison'].pop('readout_ref', None)
    recovery = {'status': 'reports_reconstructed_from_saved_training_only',
        'original_run_status_ref': {'file': str(output.parent / 'run_status.json'), 'json_path': []},
        'new_author_steps': 0, 'new_fits': 0, 'new_predictions': 0,
        'new_observations': 0, 'files_written': 0}
    return {'summary': summary, 'effects': effects, 'recovery': recovery}


def settle_saved_scene_effects(output, *, summary=None, predictions=None):
    """Refresh the small effect receipts after already-recorded predictions.

    Training mechanisms and independent terminal outcomes retain their original
    values. No training/state document or predictor is opened for this settlement.
    """
    output = Path(output).resolve()
    summary = _read(output / 'summary.json') if summary is None else summary
    predictions = _read(output / 'post97_predictions.json') if predictions is None else predictions
    effects_path = output / 'research_effects.json'
    effects = _read(effects_path)
    trials = {(row['case'], row['arm']): row for row in summary['trials']}
    prediction_arms = predictions.get('arms', [])
    by_arm = {entry['arm']: (position, entry)
              for position, entry in enumerate(prediction_arms)}
    receipt_ref = {'file': str(output / 'post97_predictions.json'), 'json_path': []}
    comparisons = {(row['left_arm'], row['right_arm']): (row, ['comparisons', position])
                   for position, row in enumerate(predictions.get('comparisons', []))}
    if len(prediction_arms) == 2:
        comparisons[(prediction_arms[0]['arm'], prediction_arms[1]['arm'])] = (
            predictions['comparison'], ['comparison'])
    for effect in ([effects] if 'cases' in effects else effects.values()):
        effect['post97_development_prediction_ref'] = deepcopy(receipt_ref)
        for pair in effect.get('cases', []):
            names = (effect['left_arm'], effect['right_arm'])
            entries, arm_refs = [], []
            for side, arm in zip(('left', 'right'), names):
                trial = trials[(pair['case'], arm)]
                pair[side]['training_only_costs'] = deepcopy(
                    trial.get('training_only_costs', trial['costs']))
                pair[side]['costs'] = deepcopy(trial['costs'])
                position, entry = by_arm.get(arm, (None, {'status': 'UNKNOWN'}))
                entries.append(entry)
                arm_refs.append({**receipt_ref, 'json_path': ['arms', position]}
                                if position is not None else None)
            left, right = pair['left']['costs'], pair['right']['costs']
            pair['right_minus_left']['target_costs'] = {
                key: right[key] - left[key] if left.get(key) is not None
                and right.get(key) is not None else None
                for key in pair['right_minus_left']['target_costs']}
            comparison, comparison_path = comparisons.get(names, ({}, None))
            pair['post97_development_predictions'] = {
                'status': 'recorded' if comparison and all(
                    entry['status'] == 'recorded' for entry in entries) else 'UNKNOWN',
                'scope': 'already_exposed_paid_history_development_prediction_not_independent_confirmation',
                'independent_confirmation': False,
                'rows': predictions.get('rows', predictions.get('expected_rows')),
                'left_mse': entries[0].get('evaluation', {}).get('mse'),
                'right_mse': entries[1].get('evaluation', {}).get('mse'),
                'changed': comparison.get('changed'),
                'right_minus_left_mse': comparison.get('right_minus_left_mse'),
                'arm_refs': arm_refs,
                'comparison_ref': ({**receipt_ref, 'json_path': comparison_path}
                                   if comparison_path is not None else None)}
    _write(effects_path, effects)
    return effects


def _complete_saved_scene_predictions(result, output, manifest, *, arms=None):
    """Apply one retained prediction receipt; never add its charge twice."""
    if not manifest.get('post97_prediction_contract'):
        return result
    from hegel_machine.cognition.method_effect_query import evaluate_saved_scene_predictions

    predictions = evaluate_saved_scene_predictions(output, manifest['source_training_ref'],
        **({'arms': tuple(arms)} if arms is not None else {}))
    result['post97_prediction_ref'] = {'file': str(output / 'post97_predictions.json'), 'json_path': []}
    result['post97_prediction_diagnostic'] = predictions['status']
    result['comparison']['post97_prediction_change'] = predictions['comparison']['changed']
    by_arm = {entry['arm']: entry for entry in predictions.get('arms', [])}
    for trial in result['trials']:
        prediction = by_arm.get(trial['arm'])
        if prediction is None:
            continue
        if trial.get('training_only_costs', trial['costs']) != prediction['costs_before']:
            raise ValueError('frozen prediction receipt must match the retained training costs')
        trial['training_only_costs'] = deepcopy(prediction['costs_before'])
        trial['costs'] = deepcopy(prediction['costs_after'])
        trial['continuation_cost'] = {
            key: trial['costs'][key] - trial['starting_costs'][key]
            for key in trial['starting_costs']}
        trial['additional_costs'] = deepcopy(trial['continuation_cost'])
        trial['post97_prediction_status'] = prediction['status']
        trial['post97_prediction_mse'] = prediction.get('evaluation', {}).get('mse')
    result['additional_costs'] = {
        key: sum(row['continuation_cost'][key] for row in result['trials'])
        for key in result['shared_original_sunk_costs_once']}
    _write(output / 'summary.json', result)
    settle_saved_scene_effects(output, summary=result, predictions=predictions)
    return result


def complete_saved_scene_reports(output):
    """Repair finished two-arm reports and finish only their frozen prediction."""
    output = Path(output).resolve()
    cached_prediction = (output / 'post97_predictions.json').is_file()
    receipt = {'status': 'started', 'new_author_steps': 0, 'new_fits': 0,
        'new_observations': 0, 'new_model_calls': 0,
        'retained_prediction_receipt_reused': cached_prediction,
        'original_run_status_changed': False, 'original_training_changed': False}
    try:
        recovered = recover_saved_scene_reports(output)
        result, effects = recovered['summary'], recovered['effects']
        _write(output / 'summary.json', result)
        _write(output / 'research_effects.json', effects)
        _update_saved_scene_effects(result, effects, output)
        manifest = _value(_read(output / 'protocol.json')['source_manifest_ref'])
        result = _complete_saved_scene_predictions(result, output, manifest)
        _write(output / 'summary.json', result)
        receipt.update(status='completed', prediction_status=result['post97_prediction_diagnostic'],
            summary_ref={'file': str(output / 'summary.json'), 'json_path': []},
            effects_ref={'file': str(output / 'research_effects.json'), 'json_path': []},
            additional_costs=deepcopy(result['additional_costs']))
    except Exception as error:
        receipt.update(status='failed_preserved', error=f'{type(error).__name__}: {error}')
        _write(output / 'completion_receipt.json', receipt)
        raise
    _write(output / 'completion_receipt.json', receipt)
    return result


def run_prepared_saved_scene(preparation, output=None, *, accepted_scene=None, workers=8):
    """Freeze and run accepted97-row branches without recreating them."""
    from hegel_machine.cognition.checkpoint_storage import clone_or_copy
    from hegel_machine.cognition.finite_native_author import export_finite_native_author
    from scripts import run_anonymous_revision_inquiry as study

    accepted_scene = (accept_prepared_saved_scene(preparation, output)
                      if accepted_scene is None else accepted_scene)
    if accepted_scene['acceptance']['status'] not in {
            'both_arms_accepted_before_scientific_execution',
            'all_registered_branches_accepted_before_scientific_execution'}:
        raise ValueError('all restored scenes must pass acceptance before freezing or execution')
    views = accepted_scene.get('branch_views')
    retained_registration = accepted_scene.get('retained_scene_registration')
    arms = tuple(accepted_scene['arms'])
    if not 1 <= workers <= 8:
        raise ValueError('saved-scene workers must stay within the registered eight-core cap')
    preparation, output, manifest, source, plan, case, carry, limits = (
        accepted_scene[key] for key in ('preparation', 'output', 'manifest', 'source',
                                      'plan', 'case', 'carry', 'limits'))
    libraries = (accepted_scene['retained_frozen_libraries'] if retained_registration
                 else _freeze_saved_scene_libraries(accepted_scene))
    snapshot = sorted({str(path.relative_to(ROOT))
        for path in (ROOT / 'src/hegel_machine').rglob('*.py')} |
        {'scripts/run_anonymous_revision_inquiry.py', 'scripts/prepare_pr3_source_successor.py',
         'scripts/run_online_inquiry_llm.py', 'scripts/run_online_inquiry_study.py'})
    for name in snapshot:
        frozen = output / 'source_at_run' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        clone_or_copy(ROOT / name, frozen)
    author_source = export_finite_native_author()
    _write(output / 'registered_author.json', {'source': author_source})
    case_id = 'retained_delay_same_C' if retained_registration else 'saved_delay_event19'
    compute_increment = (retained_registration['additional_compute_per_arm']
                         if retained_registration else 1500000)
    python_increment = (retained_registration['additional_python_seconds_per_arm']
                        if retained_registration else 1800)
    native_limit = (retained_registration['additional_method_steps_per_arm']
                    if retained_registration else plan['author']['max_steps'])
    protocol = {'scope': ('retained_C_future_library_bounded_continuation' if retained_registration else
                         'saved_scene_development_object_interventions' if views
                          else 'saved_scene_development_B_C'),
        'cases': [{**case, 'id': case_id}], 'limits': vars(limits),
        'evaluation': {'rows': 0, 'mse_threshold': limits.confirmation_mse_threshold},
        'prefix_rows': 97, 'research_organization': {'phase': 'development'},
        'source_manifest_ref': (deepcopy(retained_registration['source_manifest_ref'])
            if retained_registration else {'file': str(preparation / 'scene_manifest.json'), 'json_path': []}),
        'additional_compute_per_arm': compute_increment, 'additional_python_seconds_per_arm': python_increment,
        'original_sunk_costs_once': carry, 'original_costs_are_not_summed_per_branch': True,
        'new_observations': 0, 'terminal_rows': 0, 'author': plan['author'],
        'native_attempt_limit_per_arm': native_limit,
        'execution_workers': workers,
        'physical_cpu_groups': study._native_worker_cpu_groups(len(arms), workers),
        'runtime_snapshot': snapshot,
        'author_source_ref': {'file': str(output / 'registered_author.json'), 'json_path': ['source']},
        'runtime_entry': str(Path(__file__).resolve()), 'frozen_libraries': libraries,
        'scene_acceptance_ref': {'file': str(output / 'scene_acceptance.json'), 'json_path': []},
        'acceptance_cost_included_in_branch_budget': True,
        'post97_prediction_contract': deepcopy(manifest.get('post97_prediction_contract'))}
    if views:
        protocol.update(registered_arms=list(arms), branch_views=deepcopy(views),
            intervention_registration_ref=deepcopy(accepted_scene['intervention_registration_ref']),
            source_cost_ref=deepcopy(accepted_scene['source_cost_ref']))
    if retained_registration:
        protocol.update(retained_scene_registration_ref=deepcopy(
            accepted_scene['retained_scene_registration_ref']),
            source_cost_ref=deepcopy(plan.get('source_cost_ref')),
            common_scheduler_repair_is_PR3_effect=False,
            original_C_completed_steps_retained=accepted_scene['acceptance']['initial_method_steps'],
            additional_method_steps_per_arm=native_limit,
            frozen_prediction_compute_reserved_per_arm=48,
            identical_retained_C_starting_state=True,
            prior_consumed_candidates_and_scores_retained_in_both_arms=True)
    _write(output / 'protocol.json', protocol)
    def run_arm(arm):
        retained = accepted_scene['arms'][arm]
        agent, lab, previous = retained['agent'], retained['lab'], retained['previous']
        workspace = agent.workspace
        before = deepcopy(retained['starting_costs'])
        agent._save(plan['author']['path'], author_source)
        checkpoint = lambda data: study.write_json(workspace / 'active.json', data)
        if retained_registration:
            from dataclasses import replace

            total_config = agent.config
            agent.config = replace(total_config, compute_budget=total_config.compute_budget - 48)
            try:
                outcome = _source_research_steps(agent, lab, plan, checkpoint,
                    native_attempt_limit=native_limit)
            finally:
                agent.config = total_config
        else:
            outcome = _source_research_steps(agent, lab, plan, checkpoint,
                native_attempt_limit=native_limit)
        data = agent.result(outcome['status'])
        delta = {key: data['costs'][key] - before[key] for key in before}
        if delta['observation_cost'] or data.get('fresh_confirmation'):
            raise ValueError('same-paid development diagnostic acquired new observations')
        data.update(prior_research_cost=before, continuation_cost=delta,
            resume_origin=str(source / 'training.json'), saved_scene_cutoff={
                'event': retained.get('event_start', 20) - 1, 'rows': 97, 'case': case_id},
            original_costs_shared_not_additive=True, same_paid_source_continuation=outcome)
        if retained_registration:
            from hegel_machine.cognition.method_effect_query import retained_research_progress

            progress = retained_research_progress(agent, event_start=retained['event_start'],
                method_start=retained['method_start'])
            _write(workspace / 'research_progress_after.json', progress)
            data['retained_scene_continuation_ref'] = deepcopy(
                accepted_scene['retained_scene_registration_ref'])
        study.write_json(workspace / 'training.json', data)
        # This branch's consumer needs the compact scientific readout only;
        # the complete, already persisted training stays behind result_ref.
        return _saved_scene_arm_result(arm, data, workspace,
            manifest['library_views']['new_ids'],
            retained['validation']['acceptance']['new_costs'], agent.candidate_paths)
    completed = study._native_process_trials(run_arm, arms, workers=workers)
    results = [{key: value for key, value in row.items() if key != 'result'} for row in completed]
    result = _saved_scene_summary(results, carry)
    if retained_registration:
        result.update(scope='same_retained_C_future_old8_full18_bounded_continuation',
            retained_scene_registration_ref=deepcopy(accepted_scene['retained_scene_registration_ref']),
            common_scheduler_repair_is_PR3_effect=False,
            current_parent_and_previous_scientific_candidates_shared=True)
    if views:
        result.update(scope='registered_saved_scene_object_keep_remove_not_independent_D3',
            registered_arms=list(arms), branch_views=deepcopy(views),
            intervention_registration_ref=deepcopy(accepted_scene['intervention_registration_ref']))
    _write(output / 'summary.json', result)
    from hegel_machine.cognition.method_effect_query import registered_successor_effect
    effects = ({arm: registered_successor_effect(output, left_arm='keep', right_arm=arm,
                    source_cost_ref=accepted_scene['source_cost_ref'],
                    intervention_ref=views[arm]['manifest_ref'])
                for arm in arms if arm != 'keep'} if views else
               registered_successor_effect(output))
    _write(output / 'research_effects.json', effects)
    if views:
        result['intervention_effects_ref'] = {'file': str(output / 'research_effects.json'), 'json_path': []}
    else:
        _update_saved_scene_effects(result, effects, output)
    _write(output / 'summary.json', result)
    return _complete_saved_scene_predictions(result, output, manifest,
        arms=arms if views else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--stage', choices=('source', 'successor', 'saved-scene-run', 'registered-source-run', 'frozen-parent-evaluation', 'representation-research', 'recursive-preparation', 'recursive-revision'), default='source')
    arguments = parser.parse_args()
    if arguments.stage in {'recursive-preparation', 'recursive-revision'}:
        from hegel_machine.cognition.registered_source_control import (
            prepare_registered_parent_recursion, run_registered_parent_recursion)
        if arguments.resume_from is not None:
            parser.error('recursive registration declares its exact retained source in --spec')
        if arguments.stage == 'recursive-preparation':
            result = prepare_registered_parent_recursion(
                _read(arguments.spec), arguments.output)['acceptance']
            print(json.dumps({'status': result['status'],
                'acceptance': str(arguments.output / 'restoration_acceptance.json')}))
        else:
            from hegel_machine.cognition.experiment_runtime import run_process_trials
            result = run_process_trials(lambda unused: run_registered_parent_recursion(
                _read(arguments.spec), arguments.output), [None], 8)[0]
            print(json.dumps({'status': result['status'],
                'summary': str(arguments.output / 'summary.json')}))
        return
    if arguments.stage == 'representation-research':
        from hegel_machine.cognition.experiment_runtime import run_process_trials
        from hegel_machine.cognition.registered_source_control import run_registered_representation_research
        if arguments.resume_from is not None:
            parser.error('representation development uses its registered fresh output')
        result = run_process_trials(lambda unused: run_registered_representation_research(
            _read(arguments.spec), arguments.output), [None], 8)[0]
        print(json.dumps({'status': result['status'], 'summary': str(arguments.output/'summary.json')}))
        return
    if arguments.stage == 'frozen-parent-evaluation':
        from hegel_machine.cognition.registered_source_control import run_frozen_parent_evaluation
        if arguments.resume_from is not None:
            parser.error('frozen parent evaluation requires a fresh output, without --resume-from')
        result = run_frozen_parent_evaluation(_read(arguments.spec), arguments.output, workers=8)
        print(json.dumps({'status': result['status'], 'summary': str(arguments.output/'summary.json')}))
        return
    if arguments.stage == 'registered-source-run':
        from hegel_machine.cognition.registered_source_control import run_registered_control
        if arguments.resume_from is not None:
            from hegel_machine.cognition.registered_source_control import resume_registered_control
            result = resume_registered_control(_read(arguments.spec), arguments.resume_from,
                arguments.output, workers=8)
        else:
            result = run_registered_control(_read(arguments.spec), arguments.output, workers=8)
        print(json.dumps({'status': result['status'], 'summary': str(arguments.output/'summary.json')}))
        return
    if arguments.stage == 'saved-scene-run':
        result = run_prepared_saved_scene(arguments.spec.parent, arguments.output)
        print(json.dumps({'status': result['status'], 'summary': str(arguments.output / 'summary.json')}))
        return
    result = (prepare_frozen_successor(arguments.spec, arguments.output) if arguments.stage == 'successor'
              else prepare(arguments.spec, arguments.output))
    print(json.dumps({'status': result['status'], 'stage': arguments.stage,
        'baseline_model_count': result.get('baseline_model_count'),
        'qualified_local_proposal_count': result.get('qualified_local_proposal_count'),
        'registration': str(arguments.output.resolve() / 'registration.json')}))


if __name__ == '__main__':
    main()
