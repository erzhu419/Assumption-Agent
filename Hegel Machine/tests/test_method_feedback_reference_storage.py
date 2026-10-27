"""Method versions share paid bindings and unchanged feedback by reference."""

from copy import deepcopy

from hegel_machine.cognition.method_feedback import _condition_outcomes, _read
from test_common_method_feedback import scene, use_and_compare, freeze_usage
from test_inquiry_revision import PublicLab


def test_new_method_condition_version_retains_paid_feedback_by_reference(tmp_path):
    agent, lab, method, parent = scene(tmp_path, useful=True)
    use_and_compare(agent, lab, parent)
    freeze_usage(agent, lab, method)
    first = deepcopy(agent.theory_consolidation['candidates'][method['id']])
    initial_refs = _read(agent, first['method_use_conditions']['outcomes_ref'])
    initial = _condition_outcomes(agent, first['method_use_conditions']['outcomes_ref'])
    assert len(initial_refs) == 1 and 'file' in initial_refs[0]
    state = _read(agent, initial_refs[0])
    assert 'evidence_basis' not in state and 'evidence_basis_ref' in state
    assert 'seed_replay_work' not in state and 'subsequent_outcome' not in state
    assert initial[0]['evidence_basis']['paid_cutoff']['rows'] == 16
    # Use the original task allowance. Method analysis did not grant a new
    # world-research budget or change this synthetic laboratory's old bills.
    agent.research_mission = {}
    paid_lab = PublicLab()
    paid_lab.paid = agent.costs['observation_cost']
    lab.execute = paid_lab.execute
    agent._execute('query', {'actions': [{'kind': 'trajectory',
        'sequence': [{'u': .2}, {'u': .4}]}], 'question': {'id': parent['id']}}, lab)
    use_and_compare(agent, lab, parent)
    assert len(agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs']) == 2
    freeze_usage(agent, lab, method)
    current = agent.theory_consolidation['candidates'][method['id']]
    refs = _read(agent, current['method_use_conditions']['outcomes_ref'])
    assert len(refs) == 2
    # The first method has now received an actual two-row future return. Its
    # changed small state gets a new reference; the old frozen decision stays.
    assert refs[0] != initial_refs[0]
    assert _condition_outcomes(agent, current['method_use_conditions']['outcomes_ref'])[0][
        'paired_future']['reference_rows'] == 2
    assert _condition_outcomes(agent, first['method_use_conditions']['outcomes_ref']) == initial
    assert _read(agent, current['previous_version_ref']) == first
    # Conditions contain references, never another copy of the seed DAGs,
    # complete paid views or cumulative source/target result snapshots.
    assert all(set(reference) == {'file', 'json_path'} for reference in refs)
    files = set((agent.workspace / 'method_feedback').iterdir())
    paid = agent.costs['compute_cost']
    repeated = freeze_usage(agent, lab, method)
    assert repeated['new_compute_cost'] == 0 and agent.costs['compute_cost'] == paid
    assert set((agent.workspace / 'method_feedback').iterdir()) == files
