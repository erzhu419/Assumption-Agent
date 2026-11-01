"""A rebound Pi1 return retains its actual program and continuation state."""

from copy import deepcopy
from pathlib import Path

import pytest

from hegel_machine.cognition.method_action_research import (
    _existing_method_execution_return, read_method_reference,
)
from hegel_machine.cognition.models import ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.transfer_partitions import (
    prepare_parent_parameter_revision, quote_transfer_partition,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_native_method_return_route import route, save
from test_native_partition_proposals import forbidden, scene
from test_parent_parameter_revisions import paid_scene
from test_transfer_partitions import plan


def executed_scene(tmp_path, *, unexplained=False):
    common, _, evidence_ref, _, _, _ = scene(tmp_path, allowance=100000)
    evidence = read_method_reference(evidence_ref, common.workspace)
    parent, rows, paid, source = paid_scene(unexplained=unexplained)
    source['transfer_partition']['second_stage'] = {
        'operation': 'resolve_relation_intervals', 'compute_limit': 15000,
        'trigger': 'fitted_training_with_unexplained_return'}
    source_ref = save(tmp_path / 'source_method.json', source)
    parent_ref = save(tmp_path / 'actual_parent.json', encode_entry(parent))
    bound = prepare_parent_parameter_revision(parent, source,
        source_method_ref=source_ref, parent_ref=parent_ref)
    method = bound['research_program']
    method_ref = save(tmp_path / 'method_instance.json', method)
    quote = quote_transfer_partition(method, rows, feedback_rows=rows[-6:])
    actual = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-10))._revise_paid_structure(
        TaskSpec('parent_return', '', tuple(rows[0].features), (), ()),
        parent, rows, SearchBudget(quote['required_compute'] + 4 * len(rows)), [],
        finalists=[parent], paid_scores={parent['candidate'].program: parent['training_loss']},
        research_action='relation_parameter_revision', research_method=method,
        fit_rows=rows, feedback_rows=rows[-6:], paid_initial=paid)
    assert actual['full_comparison'] and actual['new_claims']
    assert actual['selected']['candidate'].program == actual['new_claims'][0]['candidate'].program
    after = actual['selected']['research_program']
    save(Path(evidence['after_entry_ref']['file']), encode_entry(actual['selected']))
    # The archived pre-comparison winner is not the scientific parent to which
    # this operation was bound. Only research_parent_ref identifies that body.
    save(Path(evidence['before_entry_ref']['file']), {'research_program': plan(.2, weight=4.)})
    evidence.update(research_parent_ref=parent_ref, executed_method_ref=method_ref,
        method_execution_ref=save(tmp_path / 'native_action.json', actual['action']))
    save(Path(evidence_ref['file']), evidence)
    return common, evidence_ref, evidence, method, after, actual['action']


@pytest.mark.parametrize('unexplained', [False, True])
def test_actual_parameter_return_keeps_entire_method_and_real_completion(tmp_path, monkeypatch, unexplained):
    common, evidence_ref, evidence, method, after, action = executed_scene(
        tmp_path, unexplained=unexplained)
    research = action['research']
    assert research['new_fits'] > 0 and research['bounded_first_search_complete']
    assert research['execution_complete'] is not unexplained
    from hegel_machine.cognition import research_programs, transfer_partitions
    monkeypatch.setattr(research_programs._Fitter, 'fit', forbidden)
    monkeypatch.setattr(transfer_partitions, 'propose_transfer_partitions', forbidden)
    old_source = Path(method['parent_parameter_revision']['source_method_ref']['file']).read_text()
    reviewed = route(common, allowance=100000)['method_action_research']
    published, = reviewed['transfer_partition_proposals']
    bundle = read_method_reference(published['proposal_ref'], common.workspace)
    assert bundle['status'] == 'existing_method_execution_returned'
    assert bundle['proposal']['baseline_research_program'] == method
    assert bundle['proposal']['candidates'] == [] and bundle['required_work'] is None
    assert bundle['execution_complete'] is not unexplained
    assert bundle['execution_status'] == research['status']
    assert bundle['basis']['revision_evidence_ref'] == evidence_ref
    assert bundle['executed_method_ref'] == evidence['executed_method_ref']
    assert not bundle['source_support_inherited'] and not bundle['independent_support']
    assert bundle['paired_effect'] is None and bundle['installed_candidate'] is None
    assert reviewed['relative_labels_created'] == 0 and reviewed['independent_method_gain'] is None
    assert route(common, allowance=100000)['new_compute_cost'] == 0
    assert Path(method['parent_parameter_revision']['source_method_ref']['file']).read_text() == old_source


@pytest.mark.parametrize('change', ['parent', 'source_body', 'returned_search', 'returned_columns'])
def test_parameter_return_requires_actual_binding_and_saved_fit_result(tmp_path, change):
    common, _, evidence, method, after, action = executed_scene(tmp_path)
    after = deepcopy(after)
    if change == 'parent':
        action['research']['parent_ref'] = {'file': 'different_parent.json'}
        save(Path(evidence['method_execution_ref']['file']), action)
    elif change == 'source_body':
        source_ref = method['parent_parameter_revision']['source_method_ref']
        source = read_method_reference(source_ref, common.workspace)
        source['transfer_partition']['second_stage']['compute_limit'] += 1
        save(Path(source_ref['file']), source)
    elif change == 'returned_search':
        after['transfer_partition_search']['attempts'] += 1
    else:
        different = plan(.85)
        for key in ('graph', 'term_nodes', 'output_node', 'selection_steps'):
            after[key] = different[key]
    before = read_method_reference(evidence['before_entry_ref'], common.workspace)['research_program']
    assert _existing_method_execution_return(evidence, before, after, common.workspace) is None
