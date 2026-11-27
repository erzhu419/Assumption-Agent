import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
sweeps = s['sensor_calibration_sweeps']
x, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
S, _ = m._responses(s)
pulses = item['campaign_diagnostics']
px = np.concatenate([(S @ p['w_pulse']).reshape(-1, 4) for p in pulses])
pr = np.concatenate([p['y_pulse_raw'].reshape(-1, 4) for p in pulses])
shape = (len(pulses), len(s['mu_shifts']), -1, 4)

def features(x, mode):
    a = [np.ones(len(x)), *x.T]
    if mode == 'quadratic':
        a += [x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)]
    return np.column_stack(a)

for mode in ('affine', 'quadratic', 'rational'):
    held, pulse = [], []
    for fold in range(5):
        test = np.arange(len(x)) % 5 == fold
        a = features(x[~test], mode)
        b = np.linalg.lstsq(a, raw[~test], rcond=None)[0]
        if mode == 'rational':
            # Channel-wise saturation after the full affine map. Fit on sweeps only.
            fit = least_squares(lambda p: ((a @ p[:20].reshape(5,4)) / (1+x[~test]**2*p[20:]) - raw[~test]).ravel(), np.r_[b.ravel(),np.zeros(4)], bounds=(np.r_[np.full(20,-np.inf),np.zeros(4)], np.full(24,np.inf)), gtol=1e-12)
            b, sat = fit.x[:20].reshape(5,4), fit.x[20:]
        def predict(v):
            y = features(v, mode) @ b
            return y / (1+v**2*sat) if mode == 'rational' else y
        held.append(np.sqrt(np.mean((predict(x[test])-raw[test])**2,axis=0)).max())
        pulse.append(np.sqrt(np.mean((predict(px)-pr).reshape(shape)**2,axis=2)).max())
    print(mode, 'worst held-out sweep RMS', max(held), 'worst pulse/shift/channel RMS in raw units', max(pulse))

out = m.solve_batch(s)
Z = s['Z_nuisance']
Pz = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
r = m._calibrate(s)-S@out['weights']
K = len(s['mu_shifts'])
print('observation RMS by shift before and after ridge nuisance projection', np.sqrt(np.mean(r.reshape(K,-1)**2,axis=1)), np.sqrt(np.mean((r-Z@(Pz@r)).reshape(K,-1)**2,axis=1)))
