import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d, g = p['dataset'], p['campaign_diagnostics']
o = m.solve_batch(d)
tau, theta, t = o['estimated_taus'], o['hereditary_weights'], d['eval_times']
assert 0.05 < tau[0] < tau[1] < 25
_, h = m._loads(d['calibration_load_cells'], g)
x = m._features(g, g)[:len(h)]
z = g['strain_0']; y = g['strain_curves_obs']; r = g['instant_rate_obs']
rise = -np.expm1(-t[:,None]/tau)
anchor = z[:,None]+(h[:,None]*theta[:2]) @ rise.T
observed_drift = r - h*np.sum(theta[:2]/tau)
B = rise*tau - t[:,None]
def cv(folds):
    e = np.empty_like(y); f = np.empty_like(r)
    alphas=[]
    for omit in folds:
        keep = ~omit
        q = observed_drift[keep]
        a = np.linalg.lstsq((q[:,None,None]*B).reshape(-1,2), (y[keep]-anchor[keep]-q[:,None]*t).ravel(), rcond=None)[0]
        alphas.append(a)
        yy, rr = m._chamber_response(h[omit], x[omit], z[omit], t, tau, theta, a)
        e[omit] = yy-y[omit]; f[omit] = rr-r[omit]
    rmse=lambda a:np.sqrt(np.mean(a*a))
    print('pooled strain/rate',rmse(e),rmse(f),'worst fold',max(rmse(e[o]) for o in folds),max(rmse(f[o]) for o in folds),'allocations',np.min(alphas,axis=0),np.max(alphas,axis=0))
    return e,f
print('leave coupon out:'); e,f=cv([np.arange(len(h))==i for i in range(len(h))])
print('leave lab out:'); cv([g['lab_id']==lab for lab in np.unique(g['lab_id'])])
missing=np.isnan(g['coupon_features_raw']).any(axis=1)
for name, rows in [('complete',~missing),('missing',missing)]:
    print(name,'rows',sum(rows),'strain/rate RMSE',np.sqrt(np.mean(e[rows]**2)),np.sqrt(np.mean(f[rows]**2)))
small=1e-6
ys, rs = m._chamber_response(h,x,z,np.array([0.,small]),tau,theta,m._CHAMBER_FRACTIONS)
assert np.allclose(ys[:,0],z,atol=1e-12)
err=np.max(np.abs((ys[:,1]-ys[:,0])/small-rs)); print('slope finite-difference max error',err); assert err < 2e-5
