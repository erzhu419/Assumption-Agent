import importlib.util
import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
cal = s['sensor_calibration_sweeps']
x, raw = cal['ref_rosette'], cal['raw_rosette']
S, _ = m._forward(s)
pulses = item['campaign_diagnostics']
pulse_x = [(S @ p['w_pulse']).reshape(-1, 4) for p in pulses]
pulse_raw = np.stack([p['y_pulse_raw'] for p in pulses])
bank = np.asarray(s['row_bank_id']).reshape(len(s['mu_shifts']), len(s['sensor_ij']))
base = m._rosette_features

def expanded(t):
    return np.column_stack([base(t), t**2] + [t[:,i]*t[:,j]*(t[:,i]-t[:,j]) for i,j in combinations(range(4),2)] + [t[:,i]*t[:,j]*t[:,k] for i,j,k in combinations(range(4),3)])

def fit(kind, xx, yy):
    if kind == 'current':
        b = m._fit_rosette(xx, yy)
        return lambda t: base(t) @ b
    if kind == 'expanded':
        b = np.linalg.lstsq(expanded(xx), yy, rcond=None)[0]
        return lambda t: expanded(t) @ b
    d = base(xx)
    def rational_channel(ch):
        def q(t):
            return np.column_stack((t[:,ch]**2, (t[:,ch]-t[:,(ch+1)%4])**2))
        qq = q(xx)
        def design(a):
            return d / (1 + qq @ a)[:,None]
        def project(a):
            dd = design(a)
            return dd @ np.linalg.lstsq(dd, yy[:,ch], rcond=None)[0] - yy[:,ch]
        a = least_squares(project, np.zeros(2), bounds=(0., 2.), gtol=1e-12).x
        b = np.linalg.lstsq(design(a), yy[:,ch], rcond=None)[0]
        return lambda t, dd: (dd @ b) / (1 + q(t) @ a)
    channels = [rational_channel(ch) for ch in range(4)]
    def predict(t):
        dd = base(t)
        return np.column_stack([channel(t, dd) for channel in channels])
    return predict

folds = np.array_split(np.random.default_rng(21).permutation(len(x)), 5)
for kind in ('current', 'expanded', 'rational'):
    folds_rms = []
    for test in folds:
        train = np.setdiff1d(np.arange(len(x)), test)
        r = fit(kind, x[train], raw[train])(x[test]) - raw[test]
        folds_rms.append(np.sqrt(np.mean(r*r, axis=0)).max())
    pred = fit(kind, x, raw)
    r = (np.stack([pred(t).ravel() for t in pulse_x]) - pulse_raw).reshape(len(pulses), *bank.shape)
    worst = max((float(np.sqrt(np.mean(r[p,k,bank[k]==b]**2))), p, k, int(b)) for p in range(len(pulses)) for k in range(len(bank)) for b in np.unique(bank[k]))
    print(kind, 'worst held-out sweep channel RMS', max(folds_rms), 'worst raw pulse/shift/bank RMS and slice', worst, 'pooled pulse RMS', np.sqrt(np.mean(r*r)))
print('bank counts by shift:', [{int(b):int(np.count_nonzero(row==b)) for b in np.unique(row)} for row in bank])
