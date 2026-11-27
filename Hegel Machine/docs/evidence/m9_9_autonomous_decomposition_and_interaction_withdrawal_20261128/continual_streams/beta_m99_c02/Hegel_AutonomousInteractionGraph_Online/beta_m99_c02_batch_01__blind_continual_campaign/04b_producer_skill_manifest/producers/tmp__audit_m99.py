import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares

path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
bank = c['single_bank_sweeps']
single = np.count_nonzero(x, axis=1) == 1
pulses = item['campaign_diagnostics']
S, _ = m._responses(s)
px = np.concatenate([(S @ p['w_pulse']).reshape(-1, 4) for p in pulses])
py = np.concatenate([p['y_pulse_raw'].reshape(-1, 4) for p in pulses])

def fit(kind, mask):
    pred = []
    for j in range(4):
        a, b = x[mask], raw[mask, j]
        def design(data, j=j):
            z = data[:, j]
            other = np.delete(data, j, axis=1)
            if kind == 'affine':
                return np.c_[np.ones(len(z)), z]
            if kind == 'full affine':
                return np.c_[np.ones(len(z)), z, other]
            if kind == 'pairwise':
                return np.c_[np.ones(len(z)), z, other, z[:, None]*other, other[:,0]*other[:,1], other[:,0]*other[:,2], other[:,1]*other[:,2]]
            energy = np.sum((z[:, None] - other)**2, axis=1) if kind == 'rational differences' else z*z
            return np.c_[np.ones(len(z)), z, energy]
        X = design(a)
        if kind.startswith('rational'):
            def f(p, X):
                return p[0] + p[1]*X[:,1]/(1 + p[2]*X[:,2])
            coef = np.linalg.lstsq(X[:,:2], b, rcond=None)[0]
            p = least_squares(lambda p: f(p, X)-b, [*coef, 0], bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,np.inf]), gtol=1e-13).x
            pred.append(lambda data, d=design, p=p, f=f: f(p, d(data)))
        else:
            coef = np.linalg.lstsq(X, b, rcond=None)[0]
            pred.append(lambda data, d=design, coef=coef: d(data) @ coef)
    return pred
fold = np.empty(len(x), int)
for subset in [single, ~single]:
    fold[subset] = np.arange(np.count_nonzero(subset)) % 4
for kind in ('affine', 'full affine', 'pairwise', 'rational self', 'rational differences'):
    errors = np.empty_like(raw)
    for f in range(4):
        test = fold == f
        errors[test] = np.column_stack([p(x[test]) for p in fit(kind, ~test)]) - raw[test]
    all_pred = np.column_stack([p(px) for p in fit(kind, np.ones(len(x), bool))])
    pulse_rms = np.sqrt(np.mean((all_pred-py).reshape(len(pulses),len(s['mu_shifts']),-1)**2,axis=2))
    bank_ref = np.zeros((len(bank['bank_id']),4))
    bank_ref[np.arange(len(bank_ref)),bank['bank_id']] = bank['ref_signal']
    bb = np.column_stack([p(bank_ref) for p in fit(kind, np.ones(len(x),bool))])
    bank_error = bb[np.arange(len(bb)),bank['bank_id']] - bank['raw_readout']
    print(kind, 'worst OOF RMSE by sweep regime/channel', max(np.sqrt(np.mean(errors[t]**2,axis=0)).max() for t in [single, ~single]), 'worst independent bank RMSE', max(np.sqrt(np.mean(bank_error[bank['bank_id']==j]**2)) for j in range(4)), 'worst pulse/shift raw RMSE', pulse_rms.max())
for p in pulses:
    y = m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw']))
    print('candidate calibrated pulse shift RMSE', np.sqrt(np.mean((S@p['w_pulse']-y).reshape(len(s['mu_shifts']),-1)**2,axis=1)))

out = m.solve_batch(s)
Z = s['Z_nuisance']
y = m._calibrate(s)
X, target = m._profile(S,y,s)
_, without = m._profile(S,y,dict(s,nuisance_ridge=0.0))
for scale in [1.0, -1.0]:
    perturbation = scale * (Z @ np.linalg.lstsq(Z, y-S@out['weights'], rcond=None)[0])
    _, shifted = m._profile(S,y+perturbation,s)
    ridge = s['source_ridge']*np.eye(S.shape[1]); cap,total=s['u_box'],s['total_budget']
    w,_ = m._box_simplex(X.T@X+ridge,-X.T@shifted,cap,total)
    wu,_ = m._box_simplex(S.T@S+ridge,-S.T@(y+perturbation),cap,total)
    w0,_ = m._box_simplex(S.T@S+ridge,-S.T@y,cap,total)
    print('nuisance scale',scale,'profiled weight change',np.linalg.norm(w-out['weights']),'unprofiled weight change',np.linalg.norm(wu-w0))
print('profiled vs zero-ridge observation delta',np.linalg.norm(target-without))
