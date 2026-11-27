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
x, y = cal['ref_rosette'], cal['raw_rosette']
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]

def affine(a):
    return np.c_[np.ones(len(a)), a]

def pairwise(a):
    return np.c_[affine(a), a*a, *[(a[:,i]-a[:,j])**2 for i,j in pairs]]

def rational(a, A):
    return np.c_[affine(a), a/(1+A*a*a), *[(a[:,i]-a[:,j])/(1+A*(a[:,i]-a[:,j])**2) for i,j in pairs]]

def losses(f, train, test):
    d = f(x)
    return np.sqrt(np.mean((d[test] @ np.linalg.lstsq(d[train], y[train], rcond=None)[0]-y[test])**2,axis=0))

rows = np.arange(len(x))
outer = np.array_split(rows, 5)
errs = []
for test in outer:
    tr = np.setdiff1d(rows,test)
    grid = (0.06,0.25,1.0)
    def inner(alpha):
        return max(losses(lambda a: rational(a,alpha),np.setdiff1d(tr,v),v).max() for v in np.array_split(tr,4))
    alpha = min(grid,key=inner)
    diag = np.stack([losses(lambda a: affine(a)[:,[0,i+1]],tr,test)[i] for i in range(4)])
    errs.append([diag, losses(affine,tr,test), losses(pairwise,tr,test), losses(lambda a:rational(a,alpha),tr,test)])
print('Sweep CV worst fold by channel (diagonal affine, full affine, pairwise, nested rational):')
print(np.max(errs,axis=0))

S,_ = m._forward(s)
Z = s['Z_nuisance']
Pz = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
bank = np.asarray(s['row_bank_id'])
shift = np.repeat(np.arange(len(s['mu_shifts'])),len(s['sensor_ij']))
worst = np.zeros(2)
for p in item['campaign_diagnostics']:
    r = m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) - S@p['w_pulse']
    raw_and_profiled = np.stack((r,r-Z@(Pz@r)))
    slices = np.array([np.sqrt(np.mean(raw_and_profiled[:,(shift==k)&(bank==b)]**2,axis=1)) for k in np.unique(shift) for b in np.unique(bank) if np.any((shift==k)&(bank==b))])
    worst = np.maximum(worst,slices.max(axis=0))
w = m.solve_batch(s)['weights']
print('Worst actual pulse/shift/bank RMSE (raw, ridge-profiled):', worst)
print('Budget residual:',abs(w.sum()-s['total_budget']))
assert worst[0]<3e-6 and worst[1]<3e-6, worst
