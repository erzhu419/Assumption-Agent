import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _axis_operator, _forward

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)

sweep_rmse = np.zeros((2, 4))
pulse_rmse = np.zeros((2, 4))
max_row_error = max_gap = 0.
for item in items:
    st = item['study']
    for axis in 'xy':
        k, v = st['kappa_'+axis], st['vel_'+axis]
        h = st['L'+axis] / (len(k)-1)
        A = _axis_operator(k, v, st['L'+axis]).toarray()
        for i in (0, len(k)-1):
            row = np.zeros(len(k))
            for d in (-1, 1):
                j = (i+d) % len(k)
                flux = (k[i]+k[j])/(2*h*h) + max(0, -d*v[i])/h
                row[i] += flux
                row[j] -= flux
            max_row_error = max(max_row_error, abs(A[i]-row).max())
    S, _ = _forward(st)
    out = solve_batch(st)
    cert = out['kkt_certificate']
    max_gap = max(max_gap, abs(cert['primal_objective']-cert['dual_lower_bound']))
    sw = st['sensor_calibration_sweeps']
    x, y = sw['ref_rosette'], sw['raw_rosette']
    keep = np.flatnonzero(np.all(abs(y) <= 8.5, axis=1))
    for train_drop in np.array_split(keep, 5):
        train = np.setdiff1d(keep, train_drop)
        diag = np.array([np.linalg.lstsq(np.c_[np.ones(len(train)), x[train,j]], y[train,j], rcond=None)[0] for j in range(4)])
        full = np.linalg.lstsq(np.c_[np.ones(len(train)), x[train]], y[train], rcond=None)[0]
        def invert(raw, model):
            return (raw-diag[:,0])/diag[:,1] if model == 0 else np.linalg.solve(full[1:].T, (raw-full[0]).T).T
        for model in range(2):
            err = invert(y[train_drop], model) - x[train_drop]
            sweep_rmse[model] = np.maximum(sweep_rmse[model], np.sqrt(np.mean(err**2, axis=0)))
            for p in item.get('campaign_diagnostics') or []:
                raw = p['y_pulse_raw'].reshape(-1, 4)
                err = (invert(raw, model).ravel()-S@p['w_pulse']).reshape(len(st['mu_shifts']), -1, 4)
                pulse_rmse[model] = np.maximum(pulse_rmse[model], np.sqrt(np.mean(err**2, axis=1)).max(axis=0))
print('worst held-out sweep RMSE by bank [diagonal, cross-bank]:', sweep_rmse.tolist())
print('worst tracer RMSE by bank and shift across sweep folds [diagonal, cross-bank]:', pulse_rmse.tolist())
print('max absolute periodic boundary-row error:', max_row_error, 'max public dual gap:', max_gap)
assert max_row_error < 1e-10 and max_gap < 1e-10
assert pulse_rmse[0].max() < 3e-6
