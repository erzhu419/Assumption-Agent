import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = item['dataset']
cal = d['calibration_load_cells']
x, y = (cal[k] for k in ('true_channel_loads','bridge_raw'))
a = m.basis(x)

def clean(ii):
    ai, yi = a[ii,:5], y[ii]
    r = np.max(abs(yi-ai@np.linalg.lstsq(ai, yi, rcond=None)[0]), axis=1)
    return ii[r < 6*np.median(r)]

def forward_models(fit, test):
    use = clean(fit)
    b = np.linalg.lstsq(a[use], y[use], rcond=None)[0]
    p = np.empty((2,len(test),4))
    for j in range(4):
        cols = np.flatnonzero(abs(b[:,j]) > .004*np.max(abs(b[:,j])))
        # The full quadratic basis already contains squared pairwise sums and differences.
        # Probe saturation specifically on the recovered self-cubic bridge term.
        cubic = np.flatnonzero(m.terms[cols,j] == 3)
        assert cubic.size == 1
        def eval_model(alpha, rows):
            A = a[rows][:,cols].copy()
            A[:,cubic] /= 1 + alpha*x[rows,j,None]**2
            return A
        def fit_at(alpha):
            A = eval_model(alpha,use)
            c = np.linalg.lstsq(A,y[use,j],rcond=None)[0]
            return c,np.sum((A@c-y[use,j])**2)
        opt = minimize_scalar(lambda alpha: fit_at(alpha)[1],bounds=(0.,1.),method='bounded',options={'xatol':1e-10})
        for i,alpha in enumerate((0.,opt.x)):
            p[i,:,j] = eval_model(alpha,test)@fit_at(alpha)[0]
    return p

ids = np.arange(len(x)); clean_ids = clean(ids)
oof = np.empty((2,len(x),4)); total = np.empty(len(x)); labels = np.empty(len(x),int)
for f,idx in enumerate(np.array_split(np.random.default_rng(17).permutation(ids),5)):
    fit = np.setdiff1d(ids,idx)
    oof[:,idx] = forward_models(fit,idx)
    stress = m._stress_calibration(dict(bridge_raw=y[fit],true_channel_loads=x[fit]))
    good = idx[np.isin(idx,clean_ids)]
    total[good] = stress(y[good])-x[good].sum(axis=1)
    labels[idx] = f
for i,name in enumerate(('sparse cubic','saturating cubic challenger')):
    r = oof[i,clean_ids]-y[clean_ids]
    worst = max(np.sqrt(np.mean(r[labels[clean_ids]==f]**2,axis=0)).max() for f in range(5))
    print(name,'worst held-out channel/fold RMSE',worst)
print('calibration clean rows',len(clean_ids),'worst held-out total-stress RMSE',max(np.sqrt(np.mean(total[clean_ids[labels[clean_ids]==f]]**2)) for f in range(5)))

tr,te = d['train'],d['test']; dt,t = d['past_dt'],d['eval_times']; X,_ = m._standardize(tr,te)
S = m._stress_calibration(cal); M = m._design(tr,X,S,dt,t)
Y = tr['strain_curves_obs'] - tr['strain_0'][:,None]; V = tr['instant_rate_obs']
for lab in np.unique(tr['lab_id']):
    test = tr['lab_id']==lab
    def sliced(taus):
        A,B=M(taus); return A[~test],B[~test]
    taus, theta = m._fit(sliced,Y[~test],V[~test])
    A,B=M(taus); print('benchtop held-out lab',lab,'taus',taus,'p90 early strain / late strain / rate absolute error',*[round(np.quantile(abs(r),.9),6) for r in ((A[test]@theta-Y[test])[:,:4],(A[test]@theta-Y[test])[:,-4:],B[test]@theta-V[test])])

out=m.solve_batch(d)
shifted=dict(d)
for key in ('train','test'):
    s=dict(d[key]); lab=s['lab_id'][:,None]; scale=.7+.32*lab; bias=-3.+1.4*lab
    for field in ('coupon_features_raw','ref_coupons'):
        s[field]=s[field]*scale+bias
    shifted[key]=s
altered=m.solve_batch(shifted)
err={k:float(np.max(abs(out[k]-altered[k]))) for k in out}
print('lab-specific affine feature/panel gauge max differences',err)
assert max(err.values()) < 1e-5
