from copy import deepcopy
import json

from hegel_machine.cognition.common_object_research import reconstruct_partial_role_source
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.object_induction import assess_position_experiment, position_owner_predictions
from hegel_machine.cognition.object_interventions import plan_position_owner_followup
from test_paid_position_planner import scene


def measured_stages():
    spec, model, rows, schemas, record = scene()
    model['model_id'] = 'test_apparatus'
    base = rows[1]['controls']
    moved = {**base, 'xb': -1.}
    square = [base, {**base, 'qb': -1.}, moved, {**moved, 'qb': -1.}]
    first = dict(field='xb', position_control='xb', excited_quantity='qb', quantity_control='qb',
                 candidate_owners=model['dynamic_fields'], alternative_owners=['qb', 'qc'],
                 corners=square, baseline_ref={'probe_id': 'eligible', 'position': 0},
                 owner_predictions=position_owner_predictions(model['dynamic_fields'], 'qb', model['dynamic_fields']))
    rows.extend(record(name, point) for name, point in zip(['old-q', 'old-x', 'old-xq'], square[1:]))
    rows[-1]['after']['qc'] += 1.
    before = assess_position_experiment(first, model, rows, SearchBudget(200))
    model['position_candidates']['xb']['quantities'] = before['retained_owners']
    second = plan_position_owner_followup(spec, model, rows, SearchBudget(200), previous_plan=first,
                                           previous_result=before, schemas=schemas, remaining_rows=2)
    rows.extend(record(name, second['corners'][i]) for name, i in [('new-q', 1), ('new-xq', 3)])
    rows[-1]['after']['qb'] += 1.
    after = assess_position_experiment(second, model, rows, SearchBudget(200))
    return [dict(position_experiment=p, completed_relation=r) for p, r in [(first, before), (second, after)]]


def rebuild(stages, budget):
    return reconstruct_partial_role_source(previous={'position_history': stages},
        source_ref={'file': 'paid_two_stage_return.json'}, model_id='one_source', budget=budget)


def test_reconstruct_only_rebinds_the_measured_experiment_strategy():
    budget = SearchBudget(100, spent=3)
    result = rebuild(measured_stages(), budget)
    method = result['method']
    assert 0 < result['compute_cost'] == budget.spent - 3
    assert method['source_ref'] == result['source_ref']
    assert method['position_method'] == {
        'operator': 'plan_position_owner_followup', 'remaining_owner_count': 2,
        'reuse_reset_corners': [0, 2], 'excitation_role': 'outside_remaining_owners',
        'prediction_relation': 'disjoint_foreign_effects'}
    assert not method['object_assignment_transferred'] and not method['target_independent_support']
    assert 'source_role_bindings' not in method
    assert 'qb' not in str(method) and 'old-x' not in str(method)


def test_do_not_freeze_a_flat_unpaired_or_unfunded_sequence():
    stages = measured_stages()
    assert rebuild(stages[1:], SearchBudget(100))['method'] is None
    assert rebuild(stages, SearchBudget(0))['assessment']['status'] == 'position_method_compute_exhausted'
    for change in ('flat', 'different_background', 'unchanged_experiment'):
        modified = deepcopy(stages)
        plan, paid = modified[1]['position_experiment'], modified[1]['completed_relation']
        if change == 'flat':
            paid['status'] = 'position_domain_unchanged'
        elif change == 'different_background':
            paid['corner_refs'][2]['probe_id'] = 'another_reset'
        else:
            plan['excited_quantity'] = modified[0]['position_experiment']['excited_quantity']
        assert rebuild(modified, SearchBudget(100))['method'] is None


def test_source_reconstruction_follows_original_paid_references(tmp_path):
    stages = measured_stages()
    references = []
    for number, stage in enumerate(stages):
        entries = {}
        for key, value in stage.items():
            path = tmp_path / f'{number}-{key}.json'
            path.write_text(json.dumps(value))
            entries[key] = {'file': path.name}
        references.append(entries)
    source_ref = {'file': str(tmp_path / 'source.json')}
    source = {'previous': {'position_history_refs': references},
              'source_ref': source_ref, 'model_id': 'one_source'}
    (tmp_path / 'source.json').write_text(json.dumps(source))
    receipt = reconstruct_partial_role_source(**source, budget=SearchBudget(100))
    direct = rebuild(stages, SearchBudget(100))
    assert receipt['method'] == {**direct['method'], 'source_ref': source_ref}
    assert receipt['assessment'] == direct['assessment']
    assert receipt['compute_cost'] == direct['compute_cost']
