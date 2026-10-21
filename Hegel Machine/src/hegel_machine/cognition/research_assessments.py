"""Reuse exact paid fit assessments for research, without evaluating a Program."""

from copy import deepcopy
from math import isfinite
from pathlib import Path

from .language import Program
from .models import Candidate
from .research_episodes import _child_ref, _path, load_source_values
from .selection import AssessedCandidate


def research_comparison_basis(agent, question):
    """Keep the small contract and its existing file version in the route basis."""
    contract = question.get('research_comparison')
    if contract is None:
        return None
    reference = contract.get('assessment_basis_ref', {})
    source = deepcopy(reference)
    if reference.get('file'):
        path = Path(reference['file'])
        path = path if path.is_absolute() else agent.workspace / path
        source['file'] = str(path.resolve())
        try:
            stat = path.stat()
            source.update(stat_size=stat.st_size, stat_mtime_ns=stat.st_mtime_ns)
        except OSError:
            source['unavailable'] = True
    return {'contract': deepcopy(contract), 'source_version': source}


def resolve_research_comparison(agent, question, theories):
    """Resolve full MSEs on one cited scope; keep unassessed original entries.

    The returned AssessedCandidates are transient selector inputs. Their scopes
    retain addresses, never observation payloads or unobserved future scores.
    """
    from .theory_consolidation import (
        _absolute_reference, _anchor_file_references, _reference_workspace,
        library_entry_reference,
    )

    contract = question.get('research_comparison')
    if contract is None:
        return None
    explicit = list(question.get('candidate_theory_ids') or [])
    query = question.get('query_semantics', {}).get('query_expression', {})
    explicit.extend(ref['id'] for ref in query.get('claim_refs', []) if 'id' in ref)
    mandatory = list(dict.fromkeys(explicit))
    result = {'maximum': contract.get('maximum', 0), 'assessed': [],
        'mandatory_theory_ids': mandatory, 'assessment_refs': {}, 'unknown': [],
        'status': 'UNKNOWN', 'scope': 'retained_paid_fit_research_only',
        'support_updated': False, 'task_adoption_evaluated': False}
    library = agent.theory_consolidation
    documents = {}

    def unknown(theory, reason):
        if theory['id'] not in mandatory:
            mandatory.append(theory['id'])
        result['unknown'].append({'id': theory['id'], 'reason': reason})

    def same_address(left, right):
        return left.get('file') == right.get('file') and _path(left) == _path(right)

    def anchored(value):
        return _anchor_file_references(value, agent.workspace)

    try:
        if type(result['maximum']) is not int or result['maximum'] < 0:
            raise ValueError('finite_research_comparison_maximum_required')
        reference = contract.get('assessment_basis_ref')
        if not isinstance(reference, dict) or not reference.get('file'):
            raise ValueError('retained_assessment_basis_reference_required')
        reference = _absolute_reference(reference, agent.workspace)
        basis = None
        path = _path(reference)
        if len(path) >= 3 and path[-3] == 'executions' and path[-1] == 'request_basis':
            execution = library.get('executions', {}).get(path[-2])
            if execution is not None:
                actual = library_entry_reference(agent, 'executions', path[-2], 'request_basis',
                                                  documents=documents)
                if same_address(_absolute_reference(actual, agent.workspace), reference):
                    basis = anchored(execution['request_basis'])
        if basis is None:
            loaded, = load_source_values([reference], agent.workspace)
            basis = _anchor_file_references(loaded['value'], _reference_workspace(reference))
        rows = basis.get('evidence_checks')
        if ((basis.get('fitting') or {}).get('risk_quantile') != .5
                or not isinstance(rows, list) or not rows
                or any(not item.get('source_ref') or not isinstance(item.get('observation'), dict)
                       or not isinstance(item['observation'].get('features'), dict)
                       or type(item['observation'].get('target')) not in (int, float)
                       or not isfinite(item['observation']['target']) for item in rows)):
            raise ValueError('complete_paid_mse_basis_required')
        result['assessment_basis_ref'] = reference
        result['row_count'] = len(rows)
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        for theory in theories:
            unknown(theory, str(error))
        result['reason'] = str(error)
        return result

    refs = [item['source_ref'] for item in rows]
    for theory in theories:
        identifier = theory['id']
        if not theory.get('program') or theory.get('forecast_ref'):
            unknown(theory, 'opaque_candidate_has_no_Program_fit_assessment')
            continue
        indexed = agent.theory_index.get(identifier, {})
        binding = anchored(theory.get('consolidation_ref') or {})
        if (indexed.get('version') != theory.get('version') or not indexed.get('program')
                or Program.from_dict(indexed['program']) != Program.from_dict(theory['program'])
                or any(indexed.get(key) != theory.get(key)
                       for key in ('role_signature', 'output_unit', 'context_ref'))
                or anchored(indexed.get('consolidation_ref') or {}) != binding):
            unknown(theory, 'exact_registered_Program_version_required')
            continue
        transfer = next(((index, item) for index, item in reversed(list(enumerate(library.get('transfers', []))))
            if item.get('theory_ref') == {'id': identifier, 'version': theory['version']}
            and anchored(item.get('binding')) == binding), None)
        if transfer is None:
            unknown(theory, 'exact_retained_transfer_assessment_required')
            continue
        fitting = binding.get('fitting', {})
        measured = fitting.get('training_loss')
        if (fitting.get('fitted_rows') != len(rows) or fitting.get('evidence_refs') != refs
                or type(measured) not in (int, float) or not isfinite(measured) or measured < 0
                or any(item.get('context_ref') != theory.get('context_ref') for item in rows)):
            unknown(theory, 'full_same_scope_paid_fit_required')
            continue
        def matching_execution(executions):
            return next(((key, item) for key, item in executions.items()
                if anchored(item.get('request_basis', {}).get('evidence_checks')) == rows
                and (item.get('request_basis', {}).get('fitting') or {}).get('risk_quantile') == .5
                and any(saved.get('theory_id') == identifier and saved.get('version') == theory['version']
                        and anchored(saved.get('binding')) == binding
                        for saved in item.get('result', {}).get('generated_theories', []))), None)

        execution = matching_execution(library.get('executions', {}))
        transfer_ref = library_entry_reference(agent, 'transfers', transfer[0], documents=documents)
        if execution is not None:
            execution_ref = library_entry_reference(agent, 'executions', execution[0], 'request_basis',
                                                    documents=documents)
        else:
            # A delta replaces an execution key, but its older transfer still
            # addresses the physical document containing that exact result.
            source_ref = _absolute_reference({**transfer_ref, 'json_path': _path(transfer_ref)[:-2]},
                                             agent.workspace)
            location = (source_ref['file'], tuple(_path(source_ref)), source_ref.get('workspace'))
            if location not in documents:
                loaded, = load_source_values([source_ref], agent.workspace)
                documents[location] = _anchor_file_references(loaded['value'], _reference_workspace(source_ref))
            execution = matching_execution(documents[location].get('executions', {}))
            if execution is not None:
                execution_ref = _child_ref(source_ref, 'executions', execution[0], 'request_basis')
        if execution is None:
            unknown(theory, 'matching_paid_fit_execution_required')
            continue
        assessment_ref = _child_ref(transfer_ref, 'binding', 'fitting')
        scope = {'kind': 'retained_complete_paid_mse', 'metric': 'mse', 'risk_quantile': .5,
            'row_count': len(rows), 'assessment_basis_ref': deepcopy(reference),
            'assessment_ref': assessment_ref, 'execution_basis_ref': execution_ref,
            'independent_confirmation': False, 'reference_and_future_status': 'UNKNOWN'}
        result['assessed'].append(AssessedCandidate(
            Candidate(identifier, Program.from_dict(theory['program']), 'retained_paid_fit'),
            score=measured, reference_loss=None, evidence_loss=measured, fresh_loss=None,
            assessment_scope=scope))
        result['assessment_refs'][identifier] = assessment_ref
    result['status'] = 'ready' if result['assessed'] else 'UNKNOWN'
    return result
