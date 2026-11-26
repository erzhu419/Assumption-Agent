exec(open('/tmp/polymer_eval.py').read().split("print('taus predicted")[0])
tr = d['train']; dt = d['past_dt']; t = d['eval_times']; lab = tr['lab_id']; maps = m._load_cells(d['calibration_load_cells']); x, xt = m._alignment(tr, q); keep = np.isfinite(tr['coupon_features_raw']).all(axis=1)
for k in range(4):
    sel = keep & (lab != k)
    taus, w, b = m._fit({key: val[sel] for key, val in tr.items()}, x[sel], maps, dt, t)
    h, p = m._stress(q, maps); amp = m._jump(taus, h, p, q['initial_memory_states'], dt)*w
    drift = np.c_[np.ones(len(xt)),xt] @ b
    miss = ~np.isfinite(q['coupon_features_raw']).all(axis=1)
    drift[miss] = q['instant_rate_obs'][miss] - amp[miss] @ (1/taus)
    y0 = q['strain_0']; y = y0[:,None] + amp @ (-np.expm1(-t[:,None]/taus)).T + drift[:,None]*t
    outputs = (y, (y[:,-1]-y0)/np.maximum(abs(p),.5), m._crossings(y,y0,q['critical_strain'],t))
    j = q['lab_id']==k
    print('withheld responses only lab', k, 'tau max relative', max(abs(taus/g['taus']-1)), 'strain/secant/hazard RMSE', [np.sqrt(np.mean((v[j]-g[target][j])**2)) for v,target in zip(outputs,['te_clean','te_secant','te_cross'])])
print('public true and fitted gain/bias', np.c_[g['stress_gains'],g['stress_biases'],maps]); print('public theta', g['theta']); print('fitted theta', np.r_[m._fit({key: val[keep] for key,val in tr.items()},x[keep],maps,dt,t)[1:][0], m._fit({key: val[keep] for key,val in tr.items()},x[keep],maps,dt,t)[2]])
import json
history = json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))[-1]
c = d['calibration_load_cells']; nominal = (1.,0.); command = np.asarray(c['cmd_stress']); labs = np.asarray(c['lab_id'])
for k in range(4):
    prior, ok = m._affine(*m._lab_checks(history,k), nominal)
    assert ok
    idx = np.flatnonzero(labs==k); idx = idx[np.argsort(command[idx])]
    print('archive lab',k,prior,'current check endpoints', command[idx[[0,-1]]])
    for label, retained, corrupt in [('single spot',idx[:1],False),('two spots',idx[[0,-1]],False),('three checks with EMI',idx[[0,len(idx)//2,-1]],True),('full sweep with EMI',idx,True)]:
        rows = np.r_[np.flatnonzero(labs!=k),retained]; j=labs[rows]==k
        panel = {key: np.asarray(val)[rows].copy() for key,val in c.items()}
        if corrupt:
            panel['true_stress'][np.flatnonzero(j)[0]] += .4
        est = m._load_cells(panel)[k]
        error = max(abs(np.c_[command[idx], np.ones(len(idx))] @ (est-maps[k])))
        print(label, 'max stress discrepancy against full public sweep', round(error,6))
a = m._crossings(np.array([[.2,1.],[.1,.2],[1.,0.]]),np.array([0.,0.,1.]),np.array([.1,.5,1.]),np.array([.5,1.]))
assert np.allclose(a,[.25,1.,0.])
