import sys, json, pickle
from itertools import product
import numpy as np
sys.path[:0] = ['/holdout_task', '/tmp']
import compare_chamber as chamber
from _verified_upstream import calibrate_upstream
from batch_solver import solve_batch
from downstream_solver import solve_downstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    raw = pickle.load(f)[0]['study']
with open('/holdout_task/_upstream_param_payload.json') as f:
    study = calibrate_upstream(raw, json.load(f))
y, Z = study['y_obs_raw'], study['Z_nuisance']
ns, nm = len(study['sensor_ij']), len(study['mu_shifts'])
axis = chamber.axis

def interpolated_axis(k, v, length, robin, seam):
    N = axis(k, v, length, robin, 'N')
    P = axis(k, v, length, robin, 'P')
    return N + seam * (P-N)

chamber.axis = interpolated_axis
pairs = list(product((0., .5, 1., 1.5), repeat=2))
operators = {p: chamber.forward(study, p) for p in pairs}

def score(S, train, test):
    w, gamma, _ = chamber.fit(S[train], y[train], Z[train], study)
    return S[test] @ w + Z[test] @ gamma - y[test]

for name, group in [('sensor', np.tile(np.arange(ns), nm)), ('shift', np.repeat(np.arange(nm), ns))]:
    fixed, selected = np.empty(len(y)), np.empty(len(y))
    choices = []
    for g in np.unique(group):
        test = group == g
        training_labels = np.unique(group[~test])
        def inner_loss(S):
            losses = [score(S, ~test & (group != j), group == j) for j in training_labels]
            return np.mean(np.concatenate(losses)**2)
        best = min(pairs, key=lambda p: inner_loss(operators[p][0]))
        choices.append(best)
        fixed[test] = score(operators[(1.,1.)][0], ~test, test)
        selected[test] = score(operators[best][0], ~test, test)
    for label, res in [('periodic', fixed), ('inner-selected seam', selected)]:
        worst = max(np.sqrt(np.mean(res[group == g]**2)) for g in np.unique(group))
        channel_rmse = np.sqrt(np.mean(res.reshape(-1,4)**2, axis=0))
        print(name, label, 'worst', worst, 'channel_RMSE', channel_rmse)
    print(name, 'selected seam counts', {p: choices.count(p) for p in sorted(set(choices))})

out = solve_batch(raw)
cal_out = solve_downstream(study)
np.testing.assert_allclose(out['weights'], cal_out['weights'], rtol=0, atol=1e-12)
S, U = operators[(1.,1.)]
w, _, J = chamber.fit(S, y, Z, study)
np.testing.assert_allclose(out['weights'], w, rtol=0, atol=3e-6)
np.testing.assert_allclose(out['terminal_field'], np.tensordot(out['weights'], U, axes=1), rtol=0, atol=1e-12)
print('isolated entrypoint objective error', abs(out['kkt_certificate']['primal_objective'] - J))
