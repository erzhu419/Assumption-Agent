import importlib.util
import pickle
import numpy as np
from scipy.linalg import lstsq

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
S, _ = m._forward(s)

def features(v, j, family):
    a, b, d, e = v[:, [(j+i)%4 for i in (0,-1,1,2)]].T
    B = np.column_stack((np.ones(len(a)), a, b, d, a*d, a**3))
    if family == 'pair':
        return np.column_stack((B, e, (a-d)**2, (a+b)**2, a*b*d))
    if family.startswith('rational'):
        q = a if family == 'rational_self' else a-d
        return np.column_stack((B, a/(1+q*q), (a-d)/(1+q*q)))
    return B

def fit(A, z):
    w = np.ones(len(A))
    for _ in range(40):
        q = lstsq(A*w[:,None], z*w, lapack_driver='gelsy')[0]
        w = 1 / np.hypot(1, (z-A@q)/3e-5)
    return q

for family in ('six_term', 'pair', 'rational_self', 'rational_difference'):
    sweeps, tracers = [], []
    for j in range(4):
        A = features(x,j,family)
        err = np.zeros(len(x))
        pulse_err = []
        for fold in range(5):
            test = np.arange(len(x))%5 == fold
            q = fit(A[~test],y[~test,j])
            err[test] = abs(A[test]@q-y[test,j])
            for pulse in item.get('campaign_diagnostics') or []:
                v = (S@pulse['w_pulse']).reshape(-1,4)
                raw = pulse['y_pulse_raw'].reshape(-1,4)[:,j]
                r = (features(v,j,family)@q-raw).reshape(len(s['mu_shifts']),-1)
                pulse_err.append(np.sqrt(np.mean(r*r,axis=1)).max())
        sweeps.append(np.quantile(err,.9))
        tracers.extend(pulse_err)
    print(f'{family}: worst-bank out-of-fold sweep p90={max(sweeps):.6g}; worst fold/bank/pulse/shift raw RMSE={max(tracers):.6g}')

Z = s['Z_nuisance']
yc = m._calibrate(s)
cap, total, ridge = (s[k] for k in ('u_box','total_budget','source_ridge'))
w = m.solve_batch(s)['weights']
for j in np.unique(s['row_bank_id']):
    train = s['row_bank_id'] != j
    z = Z[train]
    P = np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    A = S[train]-z@(P@S[train])
    target = yc[train]-z@(P@yc[train])
    wj, _ = m._box_budget_qp(A.T@A+ridge*np.eye(len(w)), A.T@target, cap, total)
    gamma = P@(yc[train]-S[train]@wj)
    held = ~train
    err = yc[held]-S[held]@wj
    print(f'held-out bank {j}: RMSE without/with train-only nuisance={np.sqrt(np.mean(err**2)):.6g}/{np.sqrt(np.mean((err-Z[held]@gamma)**2)):.6g}; relative weight shift={np.linalg.norm(wj-w)/np.linalg.norm(w):.6g}')
