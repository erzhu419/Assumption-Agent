import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
ds = item['dataset']
a, b, d = ds['train'], ds['test'], item['campaign_diagnostics']
t, dt = ds['eval_times'], ds['past_dt']
xa, xb = m._features(a, b)
cal = m._bridge(ds['calibration_load_cells'])
s = m._prepare(a, xa, cal)
y = a['strain_curves_obs']-a['strain_0'][:,None]
r = a['instant_rate_obs']
result = m.solve_batch(ds)
tau, th = result['estimated_taus'], result['hereditary_weights']

def direct(samples, tau, th):
    past, hold, F = samples
    z = np.zeros((len(hold),2))
    for k,h in enumerate(dt):
        e = np.exp(-h/tau)
        z = e*z+(1-e)*past[:,k,None]
    g = hold[:,None]-z
    drift = F@th[2:]
    base = (g*th[:2])@(-np.expm1(-t[:,None]/tau)).T+drift[:,None]*t
    rate = g@(th[:2]/tau)+drift
    return base, rate, z, drift

base, rate, _, drift = direct(m._prepare(b, xb, cal), tau, th)
K = tau*-np.expm1(-t[:,None]/tau)-t[:,None]
np.testing.assert_allclose(result['predicted_strain'], b['strain_0'][:,None]+base+drift[:,None]*(K@m._CHAMBER_FRACTIONS), atol=1e-12, rtol=0)
np.testing.assert_allclose(result['initial_rate_pred'], rate, atol=1e-12, rtol=0)
for lab in np.unique(a['lab_id']):
    g=a['lab_id']==lab
    tt, ww=m._fit(tuple(v[~g] for v in s),dt,t,y[~g],r[~g])
    p,q,_,_=direct(tuple(v[g] for v in s),tt,ww)
    print('train held lab',lab,'tau rel movement',np.max(abs(tt/tau-1)),'weight max movement',np.max(abs(ww-th)),'strain/rate p90',np.quantile(abs(p-y[g]),.9),np.quantile(abs(q-r[g]),.9))

fields = ('coupon_features_raw','ref_coupons','lab_id','stratum')
xd,_=m._features(d,{k:d[k][:0] for k in fields})
sd=m._prepare(d,xd,cal)
base, rate, z, drift=direct(sd,tau,th)
h=sd[1]
Y=d['strain_curves_obs']-d['strain_0'][:,None]-base
R=d['instant_rate_obs']-rate
complete=np.isfinite(d['coupon_features_raw']).all(axis=1)
# Complete cases identify the law without treating imputation error as physics.
# Panel coordinates are fixed without using any diagnostic response.
models={
    'conserved': [drift],
    'pairwise stress': [drift,h-z[:,0],h-z[:,1]],
    'quadratic drift': [drift,drift**2],
}

def fit_predict(name, train, test):
    if name == 'rational drift':
        def pred(p, idx):
            f=drift[idx]/(1+p[2]*drift[idx]**2)
            return f[:,None]*(K@p[:2])
        p=least_squares(lambda p: ((pred(p,train)-Y[train])/.003).ravel(),[.56,.70,0],bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,np.inf]),loss='soft_l1').x
        return pred(p,test)
    M=np.concatenate([v[:,None,None]*K[None,:,:] for v in models[name]],axis=2)
    A=M[train].reshape(-1,M.shape[-1]); yy=Y[train].ravel()
    p=least_squares(lambda p:(A@p-yy)/.003,np.linalg.lstsq(A,yy,rcond=None)[0],loss='soft_l1').x
    return M[test]@p

for name in [*models,'rational drift']:
    E=np.empty_like(Y)
    for lab in np.unique(d['lab_id']):
        g=d['lab_id']==lab
        E[g]=fit_predict(name,complete&~g,g)-Y[g]
    slices=[(d['lab_id']==lab)&q for lab in np.unique(d['lab_id']) for q in [complete,~complete]]
    print(name,'complete/missing strain p90',*[round(float(np.quantile(abs(E[g]),.9)),6) for g in [complete,~complete]],'worst lab/completeness/horizon p90',round(max(float(np.quantile(abs(E[g,j]),.9)) for g in slices if g.any() for j in [0,len(t)//2,len(t)-1]),6))
print('diagnostic rate p90 complete/missing',*[float(np.quantile(abs(R[g]),.9)) for g in [complete,~complete]])
