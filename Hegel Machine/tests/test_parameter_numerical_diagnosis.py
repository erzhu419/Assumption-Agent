"""Production precision separates solver rejection from saved-data failures."""

import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.revision_parameter_relations import learn_parameter_relations
from test_revision_parameter_relations import CONTRACT, sources


def run(candidate, records):
    budget = SearchBudget(3000)
    receipt = learn_parameter_relations(candidate, records, budget,
        search_contract={**CONTRACT, 'tolerance': 1e-10})
    assert receipt['actual_compute_cost'] == budget.spent
    assert not receipt['scientific_support'] and not receipt['independent_confirmation']
    return receipt


def diagnosis(receipt):
    return next(attempt['numerical_diagnosis'] for attempt in receipt['attempts']
                if attempt.get('numerical_diagnosis'))


def test_production_precision_accepts_stable_witness_and_retains_original_ridge_failure(tmp_path):
    candidate, records = sources(tmp_path)
    receipt = run(candidate, records)
    result = diagnosis(receipt)
    assert result['observed_outcome'] == 'numerical_rejection_corrected'
    assert result['parameter_identifiability'] == 'full_rank_linear_design'
    assert receipt['status'] == 'completed' and len(receipt['accepted']) == 1
    assert receipt['accepted'][0]['fitting_method'] == 'stable_lstsq'
    methods = {item['method']: item for item in result['solver_checks']}
    assert methods['ridge_lstsq']['training']['max_abs_residual'] > 1e-10
    assert methods['stable_lstsq']['training']['max_abs_residual'] < 1e-10
    assert methods['stable_lstsq']['held']['max_abs_residual'] < 1e-10
    assert not result['structure_impossibility_established']


def test_stable_training_fit_does_not_hide_a_held_counterexample(tmp_path):
    candidate, records = sources(tmp_path, bad_held=True)
    receipt = run(candidate, records)
    assert receipt['accepted'] == []
    result = diagnosis(receipt)
    assert result['observed_outcome'] == 'held_out_failure'
    stable = next(item for item in result['solver_checks'] if item['method'] == 'stable_lstsq')
    assert stable['training']['passed'] and not stable['held']['passed']
    assert stable['held']['max_abs_residual'] > .24


def test_fixed_affine_structure_reports_actual_training_mismatch(tmp_path):
    candidate, records = sources(tmp_path)
    for record in records:
        value = record['bindings']['old_coefficient'].value
        new = Program('const', value=value * value + 1.)
        record['bindings']['new_coefficient'] = new
        path = tmp_path / record['after_ref']['file']
        saved = json.loads(path.read_text())
        saved['revisions'][record['after_ref']['json_path'][1]]['after'] = new.to_dict()
        path.write_text(json.dumps(saved))
    receipt = run(candidate, records)
    assert receipt['accepted'] == []
    result = diagnosis(receipt)
    assert result['observed_outcome'] == 'stable_training_mismatch'
    assert not result['structure_impossibility_established']


def test_rank_deficient_saved_row_witness_keeps_identifiability_unknown(tmp_path):
    candidate, records = sources(tmp_path, values=((1., 2., 3., 4., 5.), (1., 2., 3., 4., 5.)))
    name = 'evidence:redundant_coordinate'
    for record in records:
        record['evidence_context'] = {
            'features': {name: 2. * record['bindings']['old_coefficient'].value},
            'role_signature': {name: {'role': 'measured_failure_control',
                'unit': 'dimensionless', 'time': 'before_revision'}},
            'selection': 'first_measured_parent_failure', 'source_refs': record['failure_refs']}
    receipt = run(candidate, records)
    assert receipt['status'] == 'UNKNOWN' and receipt['accepted']
    assert all(mapping['parameter_identifiability'] == 'UNKNOWN' for mapping in receipt['accepted'])
    result = diagnosis(receipt)
    assert result['parameter_identifiability'] == 'UNKNOWN'
    assert not result['structure_impossibility_established']
    assert all(not mapping['scientific_support'] for mapping in receipt['accepted'])
