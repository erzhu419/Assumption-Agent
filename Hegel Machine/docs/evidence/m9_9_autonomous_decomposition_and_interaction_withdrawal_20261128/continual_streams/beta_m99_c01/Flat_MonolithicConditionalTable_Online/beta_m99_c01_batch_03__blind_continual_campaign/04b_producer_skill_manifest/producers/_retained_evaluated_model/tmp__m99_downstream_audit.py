import pickle
import runpy
import numpy as np
from scipy.optimize import root

m = runpy.run_path('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
design = m['_rosette_design']
D = design(x)
S, fields = m['_forward'](s)
Z = s['Z_nuisance']
rho, ridge, cap, budget = (s[k] for k in ('nuisance_ridge','source_ridge','u_box','total_budget'))
targets = s['y_obs_raw'].reshape(-1,4)
base = m['solve_batch'](s)
rows = np.arange(len(targets)*4)
rosette = (rows % len(s['sensor_ij'])) // 4
shift = rows // len(s['sensor_ij'])

def invert(B):
    out = []
    for target in targets:
        start = np.linalg.solve(B[1:5].T, target-B[0])
        res = root(lambda v: design(v[None])[0] @ B - target, start, tol=1e-10)
        assert np.max(np.abs(res.fun)) < 1e-9
        out.append(res.x)
    return np.ravel(out)

def estimate(y,tr):
    z, a, b = Z[tr], S[tr], y[tr]
    p = np.linalg.solve(z.T@z+rho*np.eye(z.shape[1]),z.T)
    T, r = a-z@(p@a), b-z@(p@b)
    w,_ = m['_box_simplex'](T.T@T+ridge*np.eye(T.shape[1]),-T.T@r,cap,budget)
    return w,p@(b-a@w)

def errors(y):
    w,_ = estimate(y,rows>=0)
    field = np.tensordot(w,fields,axes=1)
    rel = lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
    result = [rel(w,base['weights']),rel(field,base['terminal_field'])]
    for group in (rosette,shift):
        errs = []
        for g in np.unique(group):
            te = group==g
            wt,gt = estimate(y,~te)
            errs.append(np.sqrt(np.mean((S[te]@wt+Z[te]@gt-y[te])**2)))
        result.append(max(errs))
    return np.array(result)

records = {name:[] for name in ('current','sparse')}
for fold in range(5):
    train = np.arange(len(x))%5!=fold
    B = m['_fit_rosette'](x[train],raw[train])
    E = raw[train]-D[train]@B
    center = np.median(E,axis=0)
    scale = np.maximum(1.4826*np.median(abs(E-center),axis=0),1e-7)
    good = np.flatnonzero(train)[np.all(abs(E-center)<=8*scale,axis=1)]
    A = D[good]
    se = np.sqrt(np.diag(np.linalg.pinv(A.T@A)))[:,None]*scale
    C = np.zeros_like(B)
    for j in range(4):
        keep = abs(B[:,j])>8*se[:,j]
        C[keep,j] = np.linalg.lstsq(A[:,keep],raw[good,j],rcond=None)[0]
    for name,coef in (('current',B),('sparse',C)):
        records[name].append(errors(invert(coef)))
print('columns: relative weight change, relative field change, worst rosette RMSE, worst shift RMSE')
print('production', errors(m['_calibrate'](s)))
for name,r in records.items():
    print(name,'worst over withheld calibration sweeps',np.max(r,axis=0))
print('maximum paired change in held-out chamber RMSE:',np.max(abs(np.array(records['current'])[:,2:]-np.array(records['sparse'])[:,2:]),axis=0))
