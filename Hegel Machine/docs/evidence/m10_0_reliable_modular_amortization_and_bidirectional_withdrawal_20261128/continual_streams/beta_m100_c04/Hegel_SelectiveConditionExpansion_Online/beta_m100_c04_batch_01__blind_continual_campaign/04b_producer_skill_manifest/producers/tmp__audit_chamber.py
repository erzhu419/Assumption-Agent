import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
S, _ = m._forward(s)
pulses = item['campaign_diagnostics']
W = np.column_stack([p['w_pulse'] for p in pulses])
Y = np.column_stack([p['y_pulse_raw'] for p in pulses])
Z = s['Z_nuisance']
M = np.eye(len(Z)) - Z @ np.linalg.pinv(Z)
for d in 'xy':
    k = s['kappa_'+d]
    a = m._axis(k, s['vel_'+d], s['L'+d]/(len(k)-1))
    np.testing.assert_allclose(a @ np.ones(len(k)), 0, atol=1e-11, rtol=0)
open_spec = importlib.util.spec_from_file_location('archived_open', '/tmp/hegel_continual_archive/beta_m100_c04_s00_solver.py')
base = importlib.util.module_from_spec(open_spec)
open_spec.loader.exec_module(base)
So, _ = base._forward(s)
C = s['sensor_calibration_sweeps']
X, R = C['ref_rosette'], C['raw_rosette']

def design(x, quadratic):
    columns = [np.ones(len(x)), *x.T]
    if quadratic:
        columns += [x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]
    return np.column_stack(columns)

def rms(x):
    return np.sqrt(np.mean(x*x))

folds = np.array_split(np.arange(len(X)), 5)
for quadratic in (False, True):
    sweep, refl, opened, refl_proj, open_proj = [], [], [], [], []
    for fold in folds:
        keep = np.ones(len(X), dtype=bool)
        keep[fold] = False
        T = design(X[keep], quadratic)
        assert np.linalg.matrix_rank(T) == T.shape[1]
        c = np.linalg.lstsq(T, R[keep], rcond=None)[0]
        sweep.append(rms(design(X[fold], quadratic) @ c - R[fold]))
        for operator, raw_scores, proj_scores in ((S,refl,refl_proj),(So,opened,open_proj)):
            err = np.column_stack([(design(v.reshape(-1,4), quadratic) @ c).ravel() - y for v,y in zip((operator @ W).T,Y.T)])
            for sl in np.array_split(np.arange(len(Y)), len(s['mu_shifts'])):
                for channel in range(4):
                    idx = sl[sl % 4 == channel]
                    raw_scores.extend(rms(err[idx,j]) for j in range(Y.shape[1]))
            proj_scores.extend(rms(v) for v in (M @ err).T)
    print('quadratic', quadratic, 'worst held-out sweep RMS', max(sweep), 'reflect min/max pulse-shift-channel RMS', min(refl), max(refl), 'open min/max', min(opened), max(opened), 'projected pulse reflect/open worst', max(refl_proj), max(open_proj))
    if not quadratic:
        assert max(refl) < 5e-6 and max(refl) < min(opened)
        assert max(refl_proj) < 5e-6 and max(open_proj) > 1e-2

_, yp = m._project(s, S, m._calibrate(s))
delta = Z @ np.linspace(-1.,1.,Z.shape[1])
_, yq = m._project(s, S, m._calibrate(s) + delta)
G = Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1])
expected = s['nuisance_ridge'] * Z @ np.linalg.solve(G, np.linspace(-1.,1.,Z.shape[1]))
np.testing.assert_allclose(yq-yp, expected, atol=2e-12, rtol=0)
print('nuisance direction retained fraction', np.linalg.norm(yq-yp)/np.linalg.norm(delta))
