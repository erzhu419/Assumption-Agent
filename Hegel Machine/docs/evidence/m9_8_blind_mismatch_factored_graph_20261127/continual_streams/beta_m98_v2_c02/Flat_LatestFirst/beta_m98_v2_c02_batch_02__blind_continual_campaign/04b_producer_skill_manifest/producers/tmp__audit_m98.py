import importlib.util, json, pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
with open('/tmp/hegel_continual_archive/archive_apparatus_log.json') as f:
    log = json.load(f)
for entry in log:
    print('apparatus metadata', {k: v for k, v in entry.items() if np.isscalar(v)})

c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
pairs = [(i,j) for i in range(4) for j in range(i,4)]
base = np.c_[np.ones(len(x)), x]
quad = np.c_[base, np.column_stack([x[:,i]*x[:,j] for i,j in pairs])]
for name, design in [('affine', base), ('quadratic', quad)]:
    errors = np.zeros_like(raw)
    for f in range(5):
        test = np.arange(len(x)) % 5 == f
        errors[test] = design[test] @ np.linalg.lstsq(design[~test], raw[~test], rcond=None)[0] - raw[test]
    print('calibration', name, 'pooled/worst bank RMSE', np.sqrt(np.mean(errors**2)), np.max(np.sqrt(np.mean(errors**2, axis=0))))

y, Z = m._calibrate(s), s['Z_nuisance']
cap, total, ridge = (s[k] for k in ('u_box','total_budget','source_ridge'))
ns = len(s['sensor_ij'])
shift = np.arange(len(y)) // ns
bank = np.asarray(s['row_bank_id'])
ii,jj = s['sensor_ij'].T
wall = np.tile((ii == 0) | (ii == s['nx']-1) | (jj == 0) | (jj == s['ny']-1), len(s['mu_shifts']))

def evaluate(A, test):
    fit = ~test
    B, N, yy = A[fit], Z[fit], y[fit]
    P = np.linalg.solve(N.T @ N + s['nuisance_ridge']*np.eye(N.shape[1]), N.T)
    C, t = B-N@(P@B), yy-N@(P@yy)
    H, b = C.T@C + ridge*np.eye(B.shape[1]), C.T@t
    w, nu = m._box_budget_qp(H, b, cap, total)
    g = H@w-b
    lo, hi = np.maximum(g-nu,0), np.maximum(nu-g,0)
    kkt = max(abs(w.sum()-total), max(0,-w.min(),w.max()-cap), np.max(abs(w*lo)), np.max(abs((cap-w)*hi)))
    return y[test] - A[test]@w - Z[test]@(P@(yy-B@w)), kkt

for label, ax, ay in [('open/open',1,1), ('open/sealed',1,0), ('sealed/open',0,1), ('sealed/sealed',0,0)]:
    A,_ = m._forward(dict(s, robin_x=ax*s['robin_x'], robin_y=ay*s['robin_y']))
    e = np.zeros_like(y)
    kkt = 0.0
    for k in np.unique(shift):
        test = shift == k
        e[test], violation = evaluate(A,test)
        kkt = max(kkt, violation)
    rmse = lambda v: float(np.sqrt(np.mean(v*v)))
    slices = {(int(k),int(b)): rmse(e[(shift==k)&(bank==b)]) for k in np.unique(shift) for b in np.unique(bank)}
    worst = max(slices,key=slices.get)
    print(label, 'shift RMSE', [rmse(e[shift==k]) for k in np.unique(shift)], 'worst (shift,bank), RMSE', worst, slices[worst], 'wall RMSE', rmse(e[wall]), 'max KKT', kkt)
    if label == 'open/open':
        assert kkt < 1e-10
        out = m.solve_batch(s)
        assert all(np.all(np.isfinite(out[k])) for k in ('weights','terminal_field'))
        cert = out['kkt_certificate']
        assert abs(cert['primal_objective']-cert['dual_lower_bound']) < 1e-10
