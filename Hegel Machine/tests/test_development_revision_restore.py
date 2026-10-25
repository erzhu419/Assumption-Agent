"""The saved bare proposer is resumed through the existing transfer cache."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import revision_relations as revisions
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.development_revision_restore import restore_development_revision_work
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from test_question_routing import CONTEXT, ROLES, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, constant, edit_candidate, feature
from test_theory_consolidation import agent_scene


def development_scene(tmp_path):
    agent, _ = agent_scene(tmp_path)
    candidate = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {candidate['id']: candidate}}
    consolidation._save(agent)
    program = binary('mul', constant(5.), feature())
    register_theories(agent, [theory('parent', program)])
    checks = [{**item, 'context_ref': deepcopy(CONTEXT)} for item in examples(program)]
    rows_path = tmp_path / 'paid_context_fixture.json'
    rows_path.write_text(json.dumps([item['observation'] for item in checks]))
    for check in checks:
        check['source_ref']['file'] = str(rows_path)
    arguments = {'candidate_id': candidate['id'], 'theory_id': 'development_resume',
        'parent_theory_id': 'parent', 'context_ref': deepcopy(CONTEXT),
        'role_signature': deepcopy(ROLES), 'output_unit': 'V',
        'allocation': {'compute': 10000}, 'evidence_checks': checks,
        'fitting': {**FITTING, 'operations': ['refit'], 'max_structures': 1}}
    first = consolidation.transfer_theory(agent, deepcopy(arguments))
    assert len(first['generated_theories']) == 3
    pool = resolved_theories(agent)
    parent = next(item for item in pool if item['id'] == 'parent')
    proposals, programs = [], []
    for item in first['generated_theories']:
        declaration = next(theory for theory in pool if theory['id'] == item['theory_id'])
        programs.append(declaration['program'])
        binding = item['binding']
        proposals.append({'program_ref': {'file': str(tmp_path / 'development.json'),
                'json_path': ['actual_programs', len(programs) - 1]},
            'bindings': binding['components'], 'parent_binding': binding['parent'],
            'constraint_hypothesis': binding['constraint_hypothesis'],
            'fitting': binding['fitting'], **{key: declaration[key] for key in
                ('role_signature', 'output_unit', 'context_ref')}})
    checks_path = tmp_path / 'original_checks.json'
    checks_path.write_text(json.dumps(checks))
    source = {'actual_parent_id': 'parent', 'fitting_contract': arguments['fitting'],
        'full_existing_archive': {'Program_components': 0},
        'paid_evidence_checks_ref': {'file': str(checks_path), 'json_path': []},
        'actual_programs': programs,
        'actual_fitter_calls': [{'fitter_report': {'search_completed': True}},
                               {'fitter_report': {'search_completed': False}}],
        'proposal_result': {'status': 'UNKNOWN', 'search_completed': False,
            'cursor': {'layer': 3, 'offset': 8, 'fitting': {'structures': []},
                       'pending_fits': [{'indices': [0, 0], 'position': {'layer': 0, 'offset': 1}}]},
            'proposals': proposals}, 'source_consumption_compute_cap': 10000,
        'actual_compute_cost': first['new_compute_cost']}
    source_path = tmp_path / 'development.json'
    source_path.write_text(json.dumps(source))
    registration = {'material': {'all_development_failures_fit_reports_and_costs_ref':
        {'file': str(source_path), 'json_path': []}}}
    agent.theory_consolidation['transfers'] = []
    agent.theory_consolidation['executions'] = {}
    return agent, candidate, parent, arguments, pool, registration


@pytest.mark.parametrize('available', [False, True])
def test_completed_fit_and_cursor_restore_without_fit_prediction_or_support(tmp_path, monkeypatch, available):
    agent, candidate, parent, arguments, pool, registration = development_scene(tmp_path)
    if not available:
        agent.theory_consolidation['candidates'].pop(candidate['id'])
    old_index = deepcopy(agent.theory_index)
    old_costs = deepcopy(agent.costs)
    def forbidden(*args, **kwargs):
        pytest.fail('restoring actual work must not fit or predict')
    monkeypatch.setattr(revisions, '_fit_revision_binding', forbidden)
    monkeypatch.setattr(Program, 'predict', forbidden)
    receipt = restore_development_revision_work(agent, registration, candidate=candidate,
        parent=parent, components=[], transfer_arguments=arguments,
        execution_key='relation_transfer:development_resume', pool=pool)
    assert receipt['object_available'] is available
    assert receipt['actual_proposals_retained'] == 3
    assert receipt['complete_fit_batches_recovered_and_reused'] == 1
    assert receipt['partial_fit_batches_retained_without_complete_claim'] == 1
    assert receipt['cursor_layer'] == 3 and receipt['cursor_offset'] == 8
    assert receipt['pending_fit_positions'] == 1
    assert agent.theory_index == old_index and agent.costs == old_costs
    assert all(item['status'] == 'UNKNOWN' and not item['check_basis'] and not item['checks']
               for item in agent.theory_consolidation['transfers'])
    restored = consolidation.install_theory_library(agent)
    assert restored['executions']['relation_transfer:development_resume']['cursor']['offset'] == 8


def test_changed_paid_order_is_rejected_before_state_changes(tmp_path):
    agent, candidate, parent, arguments, pool, registration = development_scene(tmp_path)
    old = deepcopy(agent.theory_consolidation)
    arguments['evidence_checks'].reverse()
    with pytest.raises(ValueError, match='paid evidence changed'):
        restore_development_revision_work(agent, registration, candidate=candidate,
            parent=parent, components=[], transfer_arguments=arguments,
            execution_key='relation_transfer:development_resume', pool=pool)
    assert agent.theory_consolidation == old
