import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']; sweeps = s['sensor_calibration_sweeps']
x, y = map(np.asarray, (sweeps['ref_rosette'], sweeps['raw_rosette']))
b = m._rosette_basis(x)
coef = m._fit_rosette(x, y)
err = y - np.einsum('njk,jk->nj', b, coef)
clean = np.max(abs(err), axis=1) < 1e-4
nonzero = np.count_nonzero(x, axis=1)
for k in np.unique(nonzero):
    mask = nonzero == k
    print('sweep active channels', k, 'count', mask.sum(), 'inliers', (mask & clean).sum(), 'median row error', np.median(np.max(abs(err[mask]), axis=1)))
print('coefficients normalized by per-channel gain', coef[:, 2:] / coef[:, 1:2])
for j in range(4):
    print('bank', j, 'inlier RMSE', np.sqrt(np.mean(err[clean, j]**2)), 'inlier max', np.max(abs(err[clean, j])))
Z = np.asarray(s['Z_nuisance']); P = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
S, _ = m._forward(s)
for i, pulse in enumerate(item['campaign_diagnostics']):
    target = S @ pulse['w_pulse']; raw = np.asarray(pulse['y_pulse_raw']).reshape(-1, 4)
    pred = m._calibrate(dict(s, y_obs_raw=raw.ravel()))
    d = pred - target
    roundtrip = np.einsum('njk,jk->nj', m._rosette_basis(pred.reshape(-1,4)), coef) - raw
    per_bank = np.linalg.norm(d.reshape(-1,4), axis=0) / np.linalg.norm(target.reshape(-1,4), axis=0)
    print('pulse', i, 'worst bank relative error', max(per_bank), 'projected/unprojected residual norm ratio', np.linalg.norm(d-Z@(P@d))/np.linalg.norm(d), 'max inversion roundtrip error', np.max(abs(roundtrip)))
print('nuisance design shape', Z.shape, 'study keys', sorted(s))
