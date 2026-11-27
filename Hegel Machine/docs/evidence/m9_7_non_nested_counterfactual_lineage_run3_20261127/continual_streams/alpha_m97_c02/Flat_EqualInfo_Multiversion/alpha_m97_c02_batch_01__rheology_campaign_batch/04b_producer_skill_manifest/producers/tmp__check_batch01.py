import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open(root+'public_alpha_m97_c02_batch_01.pkl','rb'))[0]
d, gt = item['dataset'], item['gt']
train, test = d['train'], d['test']
n = len(train['lab_id'])
data = m._prepare(d)
out = m.solve_alpha_m97_c02_batch_01(d)
for k, g in [('estimated_taus','taus'), ('hereditary_weights','theta'), ('regime_mode_weights','vp_theta')]:
    print(k, 'fit', np.round(out[k],6), 'ref', gt[g], 'max abs error', round(float(np.max(np.abs(out[k]-gt[g]))),7))
past, hold = data[:2]
print('calibration max history, hold error', np.max(np.abs(past[n:]-gt['te_hist_true'])), np.max(np.abs(hold[n:]-gt['te_hold_true'])))

def errors(label, ce, re, split, plateau):
    lab, st = split['lab_id'], split['stratum']
    missing = np.isnan(split['coupon_features_raw']).any(axis=1)
    subsets = [('all', np.ones(len(lab),bool))]
    subsets += [(f'lab{i}/plateau{int(p)}', (lab==i)&(plateau==p)) for i in range(4) for p in (False,True)]
    subsets += [(f'stratum{i}',st==i) for i in range(3)]
    subsets += [('missing',missing),('complete',~missing)]
    print(label, '(slice, n, curve RMSE, rate RMSE)')
    for s, idx in subsets:
        if idx.any(): print(s, idx.sum(), *[round(float(np.sqrt(np.mean(e[idx]**2))),6) for e in (ce,re)])
    print('curve horizon RMSE', np.round(np.sqrt(np.mean(ce**2,axis=0)),6))

errors('public clean holdout', out['predicted_strain']-gt['te_clean'], out['initial_rate_pred']-gt['te_true_rate0'], test, data[4][n:] > 0)
y = train['strain_curves_obs']-train['strain_0'][:,None]
r = train['instant_rate_obs']
cerr, rerr = np.empty_like(y), np.empty_like(r)
for lab in range(4):
    held_out = train['lab_id'] == lab
    keep = ~held_out & np.isfinite(train['coupon_features_raw']).all(axis=1)
    def sub(mask): return tuple(a[:n][mask] for a in data[:5])+data[5:]
    taus, w = m._fit(sub(keep), y[keep], r[keep])
    ca, ra = m._design(*sub(held_out), taus)
    cerr[held_out], rerr[held_out] = ca@w-y[held_out], ra@w-r[held_out]
    print('lab fold',lab,'taus',np.round(taus,6))
errors('leave-one-lab-out measured', cerr, rerr, train, data[4][:n] > 0)
