import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _predict, _invert, calibrate_upstream
from batch_solver import solve_batch
p = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))
s = p[0]['study']; c = s['sensor_calibration_sweeps']; x, y = c['ref_rosette'], c['raw_rosette']
clean = np.max(abs(y), axis=1) < 5
axis = np.count_nonzero(x, axis=1) == 1
pairing = [2, 3, 0, 1]
def extra(a, kind, j):
    t, u = a[:, j], a[:, pairing[j]]
    if kind == 'other_channels':
        v, w = a[:, (j+1)%4], a[:, (j+3)%4]
        return np.column_stack((v,w,v*v,w*w,v*w))
    if kind == 'pair_difference':
        return np.column_stack((t*u,t*t*u,t*u*u,np.sin(2.5*(t-.21*u))-np.sin(2.5*t),np.sin(2.5*(t+.21*u))-np.sin(2.5*t)))
    return np.column_stack([z/(1+.6*z*z) for z in (t,u,t-.21*u,t+.21*u)] + [u*u/(1+.6*u*u)])
models = ('baseline','other_channels','pair_difference','rational')
errs = {k: [] for k in models}
base = solve_batch(s); w0 = np.asarray(base['weights']); field0 = base['terminal_field']
max_changes = np.zeros(3); gaps = []
orders = (np.arange(len(x)), np.argsort(np.max(abs(x),axis=1)))
for order in orders:
    for te in np.array_split(order, 5):
        tr = np.setdiff1d(np.arange(len(x)),te)
        f,b = _fit(x[tr], y[tr]); goodtr = tr[clean[tr]]; goodte = te[clean[te]]
        r = y[goodtr] - _predict(x[goodtr],f,b)
        pred = _predict(x[goodte],f,b)
        for m in models:
            z = pred.copy()
            if m != 'baseline':
                for j in range(4):
                    beta = np.linalg.lstsq(extra(x[goodtr],m,j),r[:,j],rcond=1e-8)[0]
                    z[:,j] += extra(x[goodte],m,j) @ beta
            e = z-y[goodte]
            errs[m].append([np.sqrt(np.mean(e**2)), *[np.max(np.abs(e[a].mean(0))) if np.any(a) else 0. for a in (axis[goodte],~axis[goodte])]])
        cc = dict(c, ref_rosette=x[tr], raw_rosette=y[tr]); out = solve_batch(dict(s,sensor_calibration_sweeps=cc)); w = np.asarray(out['weights']); cert = out['kkt_certificate']
        max_changes = np.maximum(max_changes,[np.max(abs(w-w0)),np.max(abs(out['terminal_field']-field0)),abs(w.sum()-s['total_budget'])])
        assert np.isfinite(out['terminal_field']).all() and w.min() >= -1e-10 and w.max() <= s['u_box']+1e-10
        gaps.append(abs(cert['primal_objective']-cert['dual_lower_bound']))
for m in models: print(m, 'worst [forward RMSE, axis mean bias, mixed mean bias]', np.max(errs[m],axis=0))
print('max [weight change, terminal field change, budget error]',max_changes,'max certificate gap',max(gaps))
print('public studies',len(p),'axis clean rows',sum(axis&clean),'mixed clean rows',sum(~axis&clean),'EMI rows',np.flatnonzero(~clean).tolist(),'cohort / mode fields',[k for k in s if any(z in k for z in ('batch','class','relay','meter','cohort'))])
assert max_changes[2] < 1e-10 and max(gaps) < 1e-9
