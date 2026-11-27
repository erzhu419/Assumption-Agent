import importlib.util
import pickle
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
sweeps = s['sensor_calibration_sweeps']
x, y = sweeps['ref_rosette'], sweeps['raw_rosette']
fit = m._fit_rosette
base = m.solve_batch(s)
D, _ = m._design(s)
Z = s['Z_nuisance']
ns = len(s['sensor_ij'])
groups = np.tile(np.arange(ns) // 4, len(s['mu_shifts']))
worst = np.zeros(3)
for omit in np.array_split(np.arange(len(x)), 5):
    keep = np.ones(len(x), dtype=bool)
    keep[omit] = False
    coeff = fit(x[keep], y[keep])
    with patch.object(m, '_fit_rosette', return_value=coeff):
        out = m.solve_batch(s)
        err = [np.linalg.norm(out[k]-base[k])/np.linalg.norm(base[k]) for k in ('weights', 'terminal_field')]
        pulse_err = 0.
        for pulse in item['campaign_diagnostics']:
            delta = m._calibrate(dict(s, y_obs_raw=pulse['y_pulse_raw'])) - D @ pulse['w_pulse']
            for g in np.unique(groups):
                test = groups == g
                gamma = np.linalg.lstsq(Z[~test], delta[~test], rcond=None)[0]
                pulse_err = max(pulse_err, np.sqrt(np.mean((delta[test]-Z[test]@gamma)**2)))
        worst = np.maximum(worst, [*err, pulse_err])
assert np.isfinite(worst).all()
print('Five contiguous calibration block omissions; worst relative weight change %.3g; worst relative field change %.3g; worst held-out pulse sensor-bank RMSE %.3g' % tuple(worst))
