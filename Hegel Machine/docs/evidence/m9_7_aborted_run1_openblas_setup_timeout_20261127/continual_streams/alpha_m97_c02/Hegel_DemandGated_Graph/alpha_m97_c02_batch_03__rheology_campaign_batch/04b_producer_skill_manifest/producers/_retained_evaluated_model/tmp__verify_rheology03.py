import importlib.util, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_03.pkl', 'rb'))[0]
d, g = p['dataset'], p['gt']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
past, hold, f, e0 = m._prepare(d)
dt, t = d['past_dt'], d['eval_times']
y, r = tr['strain_curves_obs'] - e0[:n,None], tr['instant_rate_obs']
rmse = lambda e: float(np.sqrt(np.mean(e**2)))
ce, re = np.empty_like(y), np.empty_like(r)
for l in range(4):
    v = tr['lab_id'] == l
    a = ~v
    tau, w = m._fit(past[:n][a], hold[:n][a], f[:n][a], dt, t, y[a], r[a])
    b, b0 = m._basis(tau, past[:n][v], hold[:n][v], f[:n][v], dt, t)
    ce[v], re[v] = b @ w - y[v], b0 @ w - r[v]
    print('held-out lab', l, 'strain MAE/median AE, rate RMSE, taus', np.mean(abs(ce[v])), np.median(abs(ce[v])), rmse(re[v]), tau)
lab, st = tr['lab_id'], tr['stratum']
cells = [(l,s,(lab==l)&(st==s)) for l,s in np.unique(np.c_[lab,st],axis=0)]
print('worst held-out lab/stratum median abs strain and rate RMSE', max((np.median(abs(ce[v])),l,s) for l,s,v in cells), max((rmse(re[v]),l,s) for l,s,v in cells))
o = m.solve_alpha_m97_c02_batch_03(d)
ec, er = o['predicted_strain']-g['te_clean'],o['initial_rate_pred']-g['te_true_rate0']
print('worst clean-test horizon RMSE', max(rmse(ec[:,i]) for i in range(len(t))))
for label, v in [('positive',hold[n:]>0),('negative',hold[n:]<0),('large hold',abs(hold[n:])>d['crit_stress']),('missing coupon',np.isnan(te['coupon_features_raw']).any(axis=1))]:
    print(label, int(v.sum()), rmse(ec[v]), rmse(er[v]))
cal = d['calibration_load_cells']
base = m._calibrate(cal)
for l in range(4):
    ch = np.flatnonzero(cal['lab_id']==l)
    i = ch[len(ch)//2]
    for spike in (-12.,12.):
        changed = {k:v.copy() for k,v in cal.items()}
        changed['true_stress'][i] += spike
        assert np.max(abs(m._calibrate(changed)-base)) < 3e-4
        keep = (cal['lab_id']!=l) | np.isin(np.arange(len(cal['lab_id'])),ch[[0,4,9]])
        changed = {k:v[keep].copy() for k,v in cal.items()}
        spike_row = np.flatnonzero(changed['lab_id']==l)[1]
        changed['true_stress'][spike_row] += spike
        assert np.max(abs(m._calibrate(changed)[l]-m._archived_calibration(l))) < 3e-4
    keep = (cal['lab_id']!=l) | (np.arange(len(cal['lab_id']))==i)
    assert np.allclose(m._calibrate({k:v[keep] for k,v in cal.items()})[l],m._archived_calibration(l))
print('all four channels pass single-row fallback and injected EMI tests')
# Independent recurrence and near-zero slope check for basis ordering and sign.
tau = o['estimated_taus']
w = np.r_[o['hereditary_weights'],o['regime_mode_weights']]
z = np.zeros((len(hold),2))
for j,h in enumerate(dt):
    decay = np.exp(-h/tau)
    z = z*decay + past[:,j,None]*(1-decay)
gap = hold[:,None]-z
v = f @ w[2:7] + w[7]*np.sign(hold)*np.sqrt(abs(hold))
direct = (gap/tau) @ w[:2] + v + w[8]*(gap[:,0]-gap[:,1])/tau[0]
np.testing.assert_allclose(o['initial_rate_pred'],direct[n:],atol=1e-12)
b,_ = m._basis(tau,past,hold,f,dt,np.array([0.,1e-7]))
np.testing.assert_allclose((b[:,1]@w-b[:,0]@w)/1e-7,direct,rtol=0,atol=1e-6)
print('direct memory recurrence and strain derivative agree')
