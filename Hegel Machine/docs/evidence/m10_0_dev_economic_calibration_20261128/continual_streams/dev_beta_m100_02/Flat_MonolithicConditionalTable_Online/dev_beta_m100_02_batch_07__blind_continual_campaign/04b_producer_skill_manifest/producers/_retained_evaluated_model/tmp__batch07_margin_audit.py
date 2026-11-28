import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
cal = s['sensor_calibration_sweeps']
ref, raw = cal['ref_rosette'], cal['raw_rosette']
si = cal['single_bank_sweeps']
bank = si['bank_id']
signal = si['ref_signal']
readout = si['raw_readout']

def design(x, j, kind):
    p = x[:, j]
    terms = [np.ones(len(x)), p]
    if kind == 'mixing':
        terms += [x[:, k] for k in range(4) if k != j]
    if kind == 'quadratic':
        terms += [p**2]
    if kind.startswith('rational'):
        a = float(kind.split(':')[1])
        terms += [p**3 / (1 + a*p**2)]
    return np.column_stack(terms)

# Hold out entire rosettes so a shared calibration row cannot cross folds.
for kind in ('affine', 'mixing', 'quadratic', 'rational:0.25', 'rational:1'):
    err = np.empty_like(raw)
    for test in np.array_split(np.random.default_rng(93).permutation(len(ref)), 5):
        train = np.ones(len(ref), dtype=bool)
        train[test] = False
        for j in range(4):
            take = bank == j
            single_ref = np.zeros((take.sum(), 4))
            single_ref[:, j] = signal[take]
            A = np.concatenate((design(ref[train], j, kind), design(single_ref, j, kind)))
            b = np.r_[raw[train, j], readout[take]]
            coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            err[test, j] = design(ref[test], j, kind) @ coeff - raw[test, j]
    print(kind, 'held-out rosette RMSE per bank', np.sqrt(np.mean(err**2, axis=0)))

S, U = m._responses(s)
Z = s['Z_nuisance']
P = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
for key in ('y_obs_raw',):
    assert np.array_equal(s['row_bank_id'], np.arange(len(s[key])) % 4)

def shift_rms(e):
    return np.sqrt(np.mean(e.reshape(len(s['mu_shifts']), -1)**2, axis=1))

pulse_err = []
for pulse in item['campaign_diagnostics']:
    y = m._calibrate(dict(s, y_obs_raw=pulse['y_pulse_raw']))
    e = S @ pulse['w_pulse'] - y
    pulse_err.append(shift_rms(e))
    print('tracer RMSE by shift, before and after nuisance projection:', shift_rms(e), shift_rms(e-Z@(P@e)))
assert np.max(pulse_err) < 3e-6

out = m.solve_batch(s)
w, c = out['weights'], out['kkt_certificate']
y = m._calibrate(s)
T = S - Z @ (P @ S)
r = y - Z @ (P @ y)
e = T @ w - r
ridge = s['source_ridge']
g = T.T @ e + ridge*w
J = (e@e + ridge*(w@w))/2
budget, upper = s['total_budget'], s['u_box']
nu = c['nu_sum']
d = g - nu
D = J - g@w + nu*budget + upper*np.minimum(d, 0).sum()
assert np.all((w >= 0) & (w <= upper)) and abs(w.sum()-budget) < 1e-12
assert np.allclose(c['weights'], w, rtol=0, atol=0)
assert np.allclose(c['lam_lower'], np.maximum(d, 0), rtol=0, atol=1e-12)
assert np.allclose(c['lam_upper'], np.maximum(-d, 0), rtol=0, atol=1e-12)
assert np.allclose(out['terminal_field'], np.einsum('i,ijk->jk', w, U), rtol=0, atol=1e-12)
assert max(abs(J-D), abs(J-c['primal_objective']), abs(D-c['dual_lower_bound'])) < 1e-11
print('production residual RMSE by shift, before and after nuisance projection:', shift_rms(S@w-y), shift_rms(e))
print('worst tracer/shift RMSE:', np.max(pulse_err), 'public primal-dual gap:', J-D)
