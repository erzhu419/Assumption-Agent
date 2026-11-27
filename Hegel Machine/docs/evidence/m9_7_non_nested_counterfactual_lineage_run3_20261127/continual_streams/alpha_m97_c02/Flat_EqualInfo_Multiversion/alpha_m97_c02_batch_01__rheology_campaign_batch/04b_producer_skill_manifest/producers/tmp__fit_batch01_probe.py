exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
from scipy.optimize import least_squares
y = train['strain_curves_obs'] - train['strain_0'][:,None]
r = train['instant_rate_obs']
complete = np.isfinite(train['coupon_features_raw']).all(1)
lab = train['lab_id']
def sub(mask): return tuple(a[:n][mask] for a in data[:5])+data[5:]
def fit(mask, variant):
    p, w = m._fit(sub(mask), y[mask], r[mask])
    if variant in ('all', 'complete'): return p, w
    yy, rr = y[mask], r[mask]
    c, a = m._design(*sub(mask), p)
    e, f = c @ w - yy, a @ w - rr
    def mad(v, axis):
        center = np.median(v, axis=axis, keepdims=True)
        return np.maximum(1e-6, 1.4826*np.median(abs(v-center),axis=axis))
    scale = mad(e, 0) if variant == 'per_horizon' else mad(e, None)
    sr = mad(f, None)
    def fun(q):
        c,a = m._design(*sub(mask),np.exp(q[:2]))
        return np.r_[((c@q[2:]-yy)/scale).ravel(), (a@q[2:]-rr)/sr]
    q = least_squares(fun,np.r_[np.log(p),w],loss='cauchy',f_scale=1.5 if variant=='tight' else 2.5,x_scale='jac',max_nfev=100).x
    return np.exp(q[:2]),q[2:]
for variant in ('all','complete','per_horizon','tight'):
    eligible = np.ones(n,bool) if variant == 'all' else complete
    pooled = []
    for j in range(4):
        tr = eligible & (lab != j)
        te = complete & (lab == j)
        p,w=fit(tr,variant)
        c,a=m._design(*sub(te),p)
        e,f=abs(c@w-y[te]),abs(a@w-r[te])
        pooled.append([np.median(e),np.median(f),np.quantile(e,.9),np.quantile(f,.9),p[1]])
    p,w=fit(eligible,variant)
    print(variant,'worst lab (median curve/rate, p90 curve/rate)',np.max(np.array(pooled)[:,:4],axis=0).round(6),'fold slow taus',np.array(pooled)[:,4].round(5),'full taus',p.round(6))
