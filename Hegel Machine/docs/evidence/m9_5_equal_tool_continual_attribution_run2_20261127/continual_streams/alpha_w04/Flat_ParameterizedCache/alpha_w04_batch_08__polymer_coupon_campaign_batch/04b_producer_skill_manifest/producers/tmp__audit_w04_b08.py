import importlib.util
import pathlib
import pickle
import numpy as np
root = pathlib.Path('/holdout_task')
name = 'alpha_w04_batch_08_solver.py'
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
m = load(root / name, 'candidate')
old = load(next(pathlib.Path('/tmp/.hegel_candidate_archive/v01_9ab5ef51').rglob(name)), 'baseline')
with (root / 'public_alpha_w04_batch_08.pkl').open('rb') as f:
    cases = pickle.load(f)
print('public cases:', len(cases))
for c in cases:
    d = c['dataset']; tr, te = d['train'], d['test']; t, dt, sy = (d[k] for k in ('eval_times','past_dt','yield_stress'))
    panel = d['calibration_load_cells']; good = m._calibrate(panel); legacy = old._calibrate(panel)
    x, xt = m._align(tr, te); complete = np.isfinite(tr['coupon_features_raw']).all(axis=1); ct = np.isfinite(te['coupon_features_raw']).all(axis=1)
    y = tr['strain_curves_obs'] - tr['strain_0'][:, None]; r = tr['instant_rate_obs']; labs = tr['lab_id']
    ident = good.copy(); ident[2:] = [[1.,0.],[1.,0.]]
    # Train on the same three labs with the same accepted calibration: each
    # challenger changes only the physical stresses of the held-out lab.
    state = m._prepare(dt, *m._stress(tr, good), x, sy)
    for lab in range(4):
        f = complete & (labs != lab); v = complete & (labs == lab); vt = ct & (te['lab_id'] == lab)
        if not f.any() or not v.any(): continue
        tf, w = m._fit(t, state[:2] + tuple(a[f] for a in state[2:]), y[f], r[f])
        for label, affine in [('mixed',good), ('legacy',legacy), ('identity_B',ident)]:
            if label != 'mixed' and np.array_equal(affine[lab],good[lab]): continue
            p = m._prepare(dt, *m._stress(tr,affine), x, sy)
            a,b = m._design(tf,t,p[:2]+tuple(z[v] for z in p[2:])); ec = a@w-y[v]; er = b@w-r[v]
            zt = m._prepare(dt,*m._stress(te,affine),xt,sy); _,bt = m._design(tf,t,zt[:2]+tuple(z[vt] for z in zt[2:]))
            rt = bt@w-te['instant_rate_obs'][vt]
            print('lab',lab,label,'train held-out n/curve RMSE/curve medAE/onset RMSE',int(v.sum()),*np.round([np.sqrt(np.mean(ec**2)),np.median(abs(ec)),np.sqrt(np.mean(er**2))],6),'test onset n/RMSE',int(vt.sum()),round(float(np.sqrt(np.mean(rt**2))),6) if rt.size else None,'signed horizon medians',np.round(np.median(ec,axis=0),4).tolist())
