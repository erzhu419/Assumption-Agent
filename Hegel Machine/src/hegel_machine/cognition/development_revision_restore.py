"""Restore an actual development proposal batch into ordinary transfer state."""

from copy import deepcopy

from .language import Program
from .models import SearchBudget
from .question_routing import resolved_theories
from .research_episodes import load_source_values
from . import revision_relations as revisions
from . import theory_consolidation as consolidation


def restore_development_revision_work(agent, registration, *, candidate, parent,
                                      components, transfer_arguments, execution_key,
                                      pool=None, persist=True):
    """Retain completed solves and continue the same ordered finite problem.

    The caller restores the source scene and registers the common development
    bank first. This routine neither installs the new object nor fits, predicts,
    scores, or inherits scientific support. Removal therefore keeps dormant
    work without making the removed object available.
    """
    source_ref = registration['material']['all_development_failures_fit_reports_and_costs_ref']
    loaded = load_source_values([source_ref], agent.workspace)[0]['value']
    source_ref = consolidation._absolute_reference(source_ref, agent.workspace)
    saved = loaded['proposal_result']
    if execution_key != 'relation_transfer:' + transfer_arguments['theory_id']:
        raise ValueError('development execution key must match the ordinary transfer')
    if candidate['id'] != transfer_arguments['candidate_id']:
        raise ValueError('development candidate does not match the transfer')
    if loaded['actual_parent_id'] != parent['id']:
        raise ValueError('development parent identity changed')
    if loaded['fitting_contract'] != transfer_arguments['fitting']:
        raise ValueError('development fitting contract changed')
    if len(components) != loaded['full_existing_archive']['Program_components']:
        raise ValueError('development cursor requires its original component roster')
    checks = consolidation._resolve_paid_checks(agent, transfer_arguments['evidence_checks'])
    original_checks = load_source_values([loaded['paid_evidence_checks_ref']], agent.workspace)[0]['value']
    if not consolidation._same_relation_work(agent, {'evidence_checks': original_checks},
                                              {'evidence_checks': transfer_arguments['evidence_checks']}):
        raise ValueError('development paid evidence changed')
    if any('observation' not in item for item in checks):
        raise ValueError('development paid observations are unavailable')

    declarations = resolved_theories(agent) if pool is None else pool
    by_program = {}
    for theory in declarations:
        if theory.get('program'):
            by_program.setdefault(Program.from_dict(theory['program']).expanded(), []).append(theory)
    programs = [Program.from_dict(item) for item in loaded['actual_programs']]
    transfers, generated = [], []
    for index, proposal in enumerate(saved['proposals']):
        program = programs[proposal['program_ref']['json_path'][-1]]
        existing = next((item for item in by_program.get(program.expanded(), [])
                         if all(item.get(key) == proposal.get(key) for key in
                                ('role_signature', 'output_unit', 'context_ref'))), None)
        if existing is None:
            raise ValueError('an actual development proposal is absent from the common archive')
        if any(proposal['parent_binding'].get(key) != value for key, value in
               (('component_id', parent['id']), ('version', parent.get('version', 1)))):
            raise ValueError('development parent version changed')
        binding = {'library_path': transfer_arguments.get('library_path', consolidation.LIBRARY_PATH),
            'candidate_id': candidate['id'], 'candidate_version': candidate['version'],
            'components': deepcopy(proposal['bindings']),
            'parent': deepcopy(proposal['parent_binding']),
            'constraint_hypothesis': proposal['constraint_hypothesis'],
            'context_ref': deepcopy(proposal['context_ref']),
            'fitting': deepcopy(proposal['fitting'])}
        identity = {'id': existing['id'], 'version': existing.get('version', 1)}
        reference = {**deepcopy(source_ref), 'json_path': [*source_ref['json_path'],
                     'proposal_result', 'proposals', index]}
        # No check was performed by the original bare proposer. Training loss
        # remains training loss; the ordinary transfer must check this binding.
        transfer = {'theory_id': existing['id'], 'theory_ref': identity,
            'binding': binding, 'status': 'UNKNOWN', 'check_basis': [],
            'checks': [], 'support_updated': False, 'independent_confirmation': False,
            'development_proposal_ref': reference}
        transfers.append(transfer)
        generated.append({'theory_id': existing['id'], 'version': identity['version'],
            'status': 'UNKNOWN', 'binding': binding, 'development_proposal_ref': reference})

    library = agent.theory_consolidation
    retained = library.setdefault('transfers', [])
    retained.extend(item for item in transfers if item not in retained)
    basis = {key: deepcopy(transfer_arguments.get(key)) for key in (
        'component_theory_ids', 'component_bindings', 'role_signature', 'output_unit',
        'context_ref', 'evidence_checks', 'allocation', 'parent_theory_id', 'fitting')}
    basis.update(library_path=transfer_arguments.get('library_path', consolidation.LIBRARY_PATH),
        candidate_id=candidate['id'], candidate_version=candidate['version'],
        parent={key: deepcopy(parent.get(key)) for key in ('id', 'version', 'source_ref', 'source_status')},
        components=[{key: deepcopy(item.get(key)) for key in
            ('id', 'version', 'source_ref', 'compatible', 'source_status')} for item in components])
    if parent.get('_contract_source'):
        basis['parent']['contract_source'] = deepcopy(parent['_contract_source'].get(
            'source_ref', parent['_contract_source']))
    execution = {'request_basis': basis, 'allowance': loaded['source_consumption_compute_cap'],
        'cursor': deepcopy(saved['cursor']), 'search_completed': saved['search_completed'],
        'result': {key: deepcopy(value) for key, value in saved.items() if key != 'proposals'},
        'development_source_ref': deepcopy(source_ref)}
    execution['result']['generated_theories'] = generated
    execution['result']['new_compute_cost'] = loaded['actual_compute_cost']
    execution['result']['execution_key'] = execution_key
    library.setdefault('executions', {})[execution_key] = execution

    # Verify the real existing fit-cache consumer and its no-evaluation reuse
    # operation, rather than accepting a count of stored report dictionaries.
    audit_budget = SearchBudget(1_000_000)
    records = consolidation._retained_revision_fits(agent, candidate, parent, audit_budget)
    complete = sum(item['fitter_report']['search_completed'] for item in loaded['actual_fitter_calls'])
    if len(records) != complete:
        raise ValueError(f'complete development fit recovery mismatch: {len(records)} != {complete}')
    index = revisions._retained_fit_index(records, audit_budget)
    for record in records:
        reused = revisions._reuse_revision_fit(record['template'], record['rows'],
            record['risk_quantile'], record, records, audit_budget, index=index)
        if reused is None or reused[2]['actual_compute_cost'] != 0:
            raise ValueError('saved complete development fit was not reusable')
    previous, _ = consolidation._shared_relation_progress(agent, transfer_arguments,
        candidate, basis, execution_key, library['executions'])
    if previous is None or previous['cursor'] != saved['cursor']:
        raise ValueError('ordinary transfer did not recover the saved development cursor')
    if persist:
        consolidation._save(agent)
    return {'status': 'restored', 'development_source_ref': source_ref,
        'execution_key': execution_key, 'object_available': candidate['id'] in library.get('candidates', {}),
        'actual_proposals_retained': len(transfers), 'actual_fitter_calls_retained': len(loaded['actual_fitter_calls']),
        'complete_fit_batches_recovered_and_reused': len(records),
        'partial_fit_batches_retained_without_complete_claim': len(loaded['actual_fitter_calls']) - complete,
        'cursor_recovered': True, 'cursor_layer': saved['cursor']['layer'],
        'cursor_offset': saved['cursor']['offset'], 'pending_fit_positions': len(saved['cursor']['pending_fits']),
        'validation_compute_units': audit_budget.spent, 'new_fits': 0, 'new_scores': 0,
        'new_observations': 0, 'new_paid_model_calls': 0, 'scientific_support_inherited': False}
