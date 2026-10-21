"""Read measured pre-revision conditions identically at learning and use sites."""

from copy import deepcopy


SELECTION = 'first_measured_parent_failure'


def evidence_context(parent, examples, roles, output_unit, checks, budget):
    """Use the first cited computed failure in its existing evidence order.

    Supplied checks are the already charged before-Program predictions. Missing
    checks use the same ordinary discrepancy evaluator. No after prediction,
    parameter, source identity or future result enters the numeric inputs.
    """
    from .revision_relations import _discrepancy, _spend

    result = {'status': 'UNKNOWN', 'selection': SELECTION, 'features': {},
              'role_signature': {}, 'source_refs': []}
    for index, example in enumerate(examples):
        saved = checks[index] if index < len(checks) else None
        _spend(budget)
        if not example.get('source_ref') or 'observation' not in example:
            continue
        check = saved if saved is not None else _discrepancy(parent, example, budget)
        if check.get('status') != 'computed' or not check.get('failed'):
            continue
        row = example['observation']
        features, contracts = {}, {}

        def add(name, value, role, unit):
            if unit is None:
                return
            _spend(budget)
            features[name] = value
            contracts[name] = {'role': role, 'unit': unit, 'time': 'before_revision'}

        for name, value in row['features'].items():
            signature = roles.get(name, {})
            if 'role' in signature and 'unit' in signature and 'time' in signature:
                add(f'evidence:feature:{name}', value,
                    f'counterexample:{signature["role"]}', signature['unit'])
        add('evidence:prediction', check['prediction'], 'counterexample_parent_prediction', output_unit)
        add('evidence:residual', row['target'] - check['prediction'],
            'counterexample_signed_residual', output_unit)
        add('evidence:history_depth', len(row.get('history', [])),
            'counterexample_history_depth', 'dimensionless')
        n = example.get('criterion', {}).get('evaluated_rows')
        if type(n) is int and n > 0:
            add('evidence:paid_rows', n, 'before_paid_evaluation_rows', 'dimensionless')
        result.update(status='ready', features=features, role_signature=contracts,
            source_refs=[deepcopy(example['source_ref'])],
            criterion=deepcopy(example.get('criterion', {})),
            prediction_scope='measured_before_Program_only')
        return result
    result['reason'] = 'computed_measured_parent_failure_required'
    return result
