"""Independent small checks, NOT a replay of the 1,446 historical programs.
The scalar elimination function is transcribed from pinned synthesis.py:_solve.
The affine ridge operation matches abstractions.py:_fit_slots' linear branch.
No hidden laboratory mechanisms, new observations, or model APIs are used.
"""
import json
import numpy as np
from pathlib import Path

COMMIT = 'ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d'

def source_solve(matrix, rhs):
    width = len(rhs)
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for index in range(width):
        pivot = max(range(index, width), key=lambda row: abs(augmented[row][index]))
        augmented[index], augmented[pivot] = augmented[pivot], augmented[index]
        divisor = augmented[index][index]
        if abs(divisor) < 1e-12:
            continue
        augmented[index] = [value / divisor for value in augmented[index]]
        for row in range(width):
            if row == index:
                continue
            factor = augmented[row][index]
            augmented[row] = [value - factor * pivot_value
                              for value, pivot_value in zip(augmented[row], augmented[index])]
    return [row[-1] for row in augmented]


def affine_case(train, held):
    train, held = np.asarray(train, dtype=float), np.asarray(held, dtype=float)
    x = np.column_stack((np.ones(train.size), train))
    z = np.column_stack((np.ones(held.size), held))
    y, yh = 2. * train + 1., 2. * held + 1.
    columns = [x[:, i].tolist() for i in range(x.shape[1])]
    gram = [[sum(a * b for a, b in zip(left, right)) for right in columns] for left in columns]
    for i in range(len(columns)):
        gram[i][i] += 1e-9
    rhs = [sum(a * b for a, b in zip(column, y.tolist())) for column in columns]
    ridge = np.asarray(source_solve(gram, rhs))
    ls = np.linalg.lstsq(x, y, rcond=None)[0]
    def record(beta):
        et = float(np.max(np.abs(x @ beta - y)))
        eh = float(np.max(np.abs(z @ beta - yh)))
        return {'coefficients': beta.tolist(), 'train_max_abs': et, 'held_max_abs': eh,
                'passes_abs_1e-10': max(et, eh) <= 1e-10,
                'passes_abs_1e-8': max(et, eh) <= 1e-8}
    return {'train': train.tolist(), 'held': held.tolist(), 'true_rule': '2*x+1',
            'rank': int(np.linalg.matrix_rank(x)), 'condition_number': float(np.linalg.cond(x)),
            'pinned_ridge': record(ridge), 'svd_lstsq_control': record(ls)}

vals = np.array([0.5053650313216267,0.5051538816938639,0.5049580761603409,0.5047678342783982])
collision = {'saved_labels': vals.tolist(), 'span': float(np.ptp(vals)),
             'minimax_max_abs_lower_bound': float(np.ptp(vals)/2),
             'minimum_mean_squared_loss_on_four_equal_inputs': float(np.mean((vals-vals.mean())**2))}

# Same current coefficient, observed input, residual, old sample count, and zero history;
# different old design curvature => different exact next least-squares coefficient.
updates=[]
for old_x in ([1.,0.],[2.,0.]):
    a=np.array(old_x); b=a.copy()
    before=float(a@b/(a@a)); new_x, new_y=1.,2.
    after=float((a@b+new_x*new_y)/(a@a+new_x**2))
    updates.append({'old_x': old_x, 'old_y': b.tolist(), 'old_count':2,
                    'before':before, 'new_x':new_x, 'new_y':new_y,
                    'residual':new_y-before*new_x,'gram':float(a@a),'after':after})

results={'scope':'small mathematical and transcribed-linear-kernel checks only', 'commit':COMMIT,
         'counts': {'sample_unknown':126,'unit_unknown':1839,'fitted':723,
                    'sum':126+1839+723,'program_outputs':2*723,
                    'unit_unknown_fraction':1839/2688},
         'historical_collision_arithmetic': collision,
         'exact_relation_control_same_as_repository_test': affine_case([1.,2.,4.,5.],[3.,6.]),
         'low_variation_exact_relation_control': affine_case([.5000,.5001,.5003,.5004],[.5002,.5005]),
         'missing_curvature_counterexample': updates,
         'gate_scales': {'world_MSE':1e-10,'world_RMSE_equivalent':1e-5,
                         'parameter_absolute_error':1e-10,
                         'not_the_same_dimension_or_error_criterion':True}}
assert results['counts']['sum']==2688
assert updates[0]['before']==updates[1]['before'] and updates[0]['after']!=updates[1]['after']
assert results['exact_relation_control_same_as_repository_test']['svd_lstsq_control']['passes_abs_1e-10']
Path(__file__).with_name('diagnostic_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False,indent=2))
