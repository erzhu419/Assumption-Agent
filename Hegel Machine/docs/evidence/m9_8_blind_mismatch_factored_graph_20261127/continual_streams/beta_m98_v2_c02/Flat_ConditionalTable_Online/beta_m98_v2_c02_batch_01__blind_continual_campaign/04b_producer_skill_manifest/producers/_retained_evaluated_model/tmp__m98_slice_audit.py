import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
x, y = (np.asarray(s['sensor_calibration_sweeps'][key]) for key in ('ref_rosette', 'raw_rosette'))
F = np.c_[np.ones(len(x)), x]
rmse = lambda e: np.sqrt(np.mean(np.square(e), axis=0))
affine, quadratic = [], []
Q = np.c_[F, *[x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)]]
for fold in np.array_split(np.arange(len(x)), 5):
    train = np.setdiff1d(np.arange(len(x)), fold)
    for design, scores in ((F, affine), (Q, quadratic)):
        c = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
        scores.append(rmse(design[fold] @ c - y[fold]))
print('worst held-out sweep fold/channel RMSE, affine / quadratic:', np.max(affine), np.max(quadratic))
S, fields = m._forward(s)
Z = s['Z_nuisance']
wall = np.any((s['sensor_ij'] == 0) | (s['sensor_ij'] == np.array([s['nx']-1,s['ny']-1])), axis=1)
n = len(wall)
worst = 0.
for j, pulse in enumerate(item['campaign_diagnostics']):
    e = m._calibrate(dict(s, y_obs_raw=pulse['y_pulse_raw'])) - S @ pulse['w_pulse']
    # Fit nuisance on other shifts; check whether it explains error in a held-out shift.
    for k in range(len(s['mu_shifts'])):
        test = np.arange(k*n, (k+1)*n)
        train = np.setdiff1d(np.arange(len(e)), test)
        gamma = np.linalg.lstsq(Z[train], e[train], rcond=None)[0]
        vals = [rmse(e[test][w]) for w in (wall, ~wall) if np.any(w)]
        corrected = [rmse((e[test]-Z[test]@gamma)[w]) for w in (wall, ~wall) if np.any(w)]
        print(f'pulse={j} shift={k} wall/interior RMSE={vals} nuisance-held-out={corrected}')
        worst = max(worst, *vals)
print('worst pulse/shift/wall slice RMSE:', worst)
assert worst < 1e-5, 'check wall stencil or calibration for the localized residual'
