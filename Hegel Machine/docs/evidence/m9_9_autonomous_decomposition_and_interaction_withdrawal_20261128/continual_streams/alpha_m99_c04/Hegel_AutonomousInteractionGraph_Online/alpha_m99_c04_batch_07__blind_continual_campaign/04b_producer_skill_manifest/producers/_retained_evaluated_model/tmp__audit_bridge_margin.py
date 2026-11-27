import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
from batch_solver import solve_batch

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    ds = pickle.load(f)[0]['dataset']
cal = ds['calibration_load_cells']
x, y = cal['true_channel_loads'], cal['bridge_raw']
axis = np.count_nonzero(x, axis=1) == 1
mixed_peak = np.flatnonzero(~axis)[np.argsort(np.max(np.abs(x[~axis] * np.roll(x[~axis], -1, axis=1)), axis=1))[-12:]]
slices = {f'axis_{i}': axis & (x[:, i] != 0) for i in range(4)}
slices['strongest_mixed'] = np.isin(np.arange(len(x)), mixed_peak)

def fit_alternative(tr, te, kind):
    nxt = np.roll(x, -1, axis=1)
    A = np.stack([np.ones_like(x), *[np.broadcast_to(x[:, i, None], x.shape) for i in range(4)], x**3, x*nxt], axis=-1)
    if kind != 'baseline':
        if kind == 'signed_pair_cubic':
            extra = [(x+nxt)**3-x**3-nxt**3, (x-nxt)**3-x**3+nxt**3]
        else:
            z = x if kind == 'rational_own' else x-nxt
            extra = [z**3/(1+a*z*z) for a in (0.25, 1.0)]
        A = np.concatenate([A, np.stack(extra, axis=-1)], axis=-1)
    coeff = up._fit_bridge(x[tr], y[tr])
    er = np.linalg.norm(up._forward(x[tr], coeff) - y[tr], axis=1)
    keep = np.flatnonzero(tr)[er < 1.]
    return np.column_stack([A[te, i] @ np.linalg.lstsq(A[keep, i], y[keep, i], rcond=None)[0] for i in range(4)])

for name, te in slices.items():
    coeff = up._fit_bridge(x[~te], y[~te])
    err = np.linalg.norm(up._forward(x[te], coeff) - y[te], axis=1)
    good = err < 1.
    truth = x[te][good]
    inv = np.max(np.abs(up._invert(y[te][good], coeff) - truth))
    comparison = {}
    for kind in ('baseline', 'signed_pair_cubic', 'rational_own', 'rational_pair_difference'):
        pred = fit_alternative(~te, te, kind)
        comparison[kind] = float(np.max(np.linalg.norm((pred - y[te])[good], axis=1)))
    print(name, 'clean/spike', (int(good.sum()), int((~good).sum())), 'inverse_max', float(inv), 'forward_max', comparison)
    assert inv < .002

coeff = up._fit_bridge(x, y)
normalized = up.calibrate_upstream(ds)
print('dataset groups', {k: list(v) for k, v in ds.items() if isinstance(v, dict) and k != 'calibration_load_cells'})
for name in ('train', 'test', 'chamber_verification_trials'):
    if ds.get(name) is None:
        continue
    for field in ('past_channel_raw', 'hold_channel_raw'):
        loads = normalized[name][field].reshape(-1, 4)
        raw = ds[name][field].reshape(-1, 4)
        jac = np.broadcast_to(coeff[1:5].T, (len(loads), 4, 4)).copy()
        j = np.arange(4)
        jac[:, j, j] += 3*coeff[5]*loads**2 + coeff[6]*np.roll(loads, -1, axis=1)
        jac[:, j, (j+1)%4] += coeff[6]*loads
        sigma_min = float(np.linalg.svd(jac, compute_uv=False)[:, -1].min())
        residual = float(np.max(np.abs(up._forward(loads, coeff) - raw)))
        print(name, field, 'min_jacobian_singular_value', sigma_min, 'roundtrip_max', residual)
        assert sigma_min > .25 and residual < 1e-9
out = solve_batch(ds)
shapes = {'estimated_taus': (2,), 'hereditary_weights': (7,), 'predicted_strain': (len(ds['test']['lab_id']), len(ds['eval_times'])), 'initial_rate_pred': (len(ds['test']['lab_id']),)}
for field, shape in shapes.items():
    a = np.asarray(out[field]); assert a.shape == shape and np.isfinite(a).all(), field
print('isolated composed outputs passed')
