import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
sweeps = s['sensor_calibration_sweeps']
x, y = sweeps['ref_rosette'], sweeps['raw_rosette']
S, _ = m._design(s)
w0 = np.asarray(m.solve_batch(s)['weights'])
worst_shift = 0.0
worst_weight_change = 0.0
for fold, omitted in enumerate(np.array_split(np.random.default_rng(2026).permutation(len(x)), 5)):
    keep = np.ones(len(x), dtype=bool)
    keep[omitted] = False
    t = dict(s, sensor_calibration_sweeps=dict(sweeps, ref_rosette=x[keep], raw_rosette=y[keep]))
    w = np.asarray(m.solve_batch(t)['weights'])
    delta = np.linalg.norm(w-w0)/np.linalg.norm(w0)
    pulse_max = 0.0
    for p in item['campaign_diagnostics']:
        e = m._calibrate(dict(t, y_obs_raw=p['y_pulse_raw'])) - S@p['w_pulse']
        pulse_max = max(pulse_max, np.sqrt(np.mean(e.reshape(len(s['mu_shifts']), -1)**2, axis=1)).max())
    worst_shift = max(worst_shift, pulse_max)
    worst_weight_change = max(worst_weight_change, delta)
    print(f'fold {fold}: worst tracer shift RMSE={pulse_max:.6g}; relative weight change={delta:.6g}')
print(f'worst transfer RMSE={worst_shift:.6g}; worst relative weight change={worst_weight_change:.6g}')
