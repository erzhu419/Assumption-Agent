import importlib.util, pickle
import numpy as np

s = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_w03_batch_06_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
e = pickle.load(open('/holdout_task/public_alpha_w03_batch_06.pkl', 'rb'))[0]
d = e['dataset']; tr = d['train']; te = d['test']; n = len(tr['lab_id']); x,h,p = m._prep(d)
t, dt, ys = d['eval_times'], d['past_dt'], d['yield_stress']
lab = tr['lab_id']; tlab = te['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
tcomplete = np.isfinite(te['coupon_features_raw']).all(axis=1)
z = tr['strain_curves_obs'] - tr['strain_0'][:,None]; r = tr['instant_rate_obs']
rng = np.random.default_rng(610)
fold = np.empty(n, int)
for j in range(4):
    for ok in (False,True):
        ids = rng.permutation(np.flatnonzero((lab==j)&(complete==ok)))
        fold[ids] = np.arange(len(ids)) % 4

def predict(fit, ix):
    est = m._fit(x[fit], h[fit], p[fit], dt, t, ys, z[fit], r[fit])
    c,a = m._design(est[0], x[ix], h[ix], p[ix], dt, t, ys)
    return c @ est[1], a @ est[1]

def slope(a):
    use = t > np.median(t)
    return np.nanmedian(a[:,use] / t[use], axis=1)

def coeff(ix,s,u):
    ok = ix & np.isfinite(s) & np.isfinite(u)
    if ok.sum() < 6: return 0.
    v = u[ok]; y = s[ok]
    return np.clip(np.mean(v*y)/max(np.mean(v*v), 1e-12),0.,1.)

def reliability(ix,s,u):
    return [coeff(ix & complete & (lab//2==wing),s,u) for wing in (0,1)] + [coeff(ix & ~complete,s,u)]

def apply(k, lab, complete, u):
    w = np.where(complete, np.asarray(k)[lab//2], k[2]); return w*np.nan_to_num(u)

def rm(a): return np.sqrt(np.nanmean(a*a))

sl=np.empty(n); inn=np.empty(n); adj=np.empty(n)
for j in range(4):
    outer=fold==j
    ss=np.full(n,np.nan); uu=np.full(n,np.nan)
    for k in range(4):
        if k == j: continue
        val=fold==k
        c,a=predict(np.flatnonzero(complete & ~outer & ~val), np.flatnonzero(val))
        ss[val]=slope(z[val]-c); uu[val]=r[val]-a
    w=reliability(~outer,ss,uu)
    c,a=predict(np.flatnonzero(complete & ~outer),np.flatnonzero(outer))
    sl[outer]=slope(z[outer]-c); inn[outer]=r[outer]-a
    adj[outer]=apply(w,lab[outer],complete[outer],inn[outer])
    print('outer blend',j,np.round(w,3))
base=np.where(complete,0.,np.nan_to_num(inn))
for tag,k in [('lab'+str(j),(lab==j)&complete) for j in range(4)]+[('missing',~complete)]:
    print('nested slope RMSE',tag,k.sum(),*[round(rm((sl-v)[k]),6) for v in [base,adj,inn]])
w=reliability(np.ones(n,bool),sl,inn)
print('final training-only blend',np.round(w,3))
c,a=predict(np.flatnonzero(complete),np.arange(n,len(x)))
u=te['instant_rate_obs']-a
g=e['gt']; ce=c+te['strain_0'][:,None]-g['te_clean']; re=a-g['te_true_rate0']
for tag,k in [('lab'+str(j),tlab==j) for j in range(4)]+[('missing',~tcomplete),('all',np.ones(len(tlab),bool))]:
    print('public',tag,[(round(rm((ce+v[:,None]*t)[k]),6),round(rm((re+v)[k]),6)) for v in [np.where(tcomplete,0.,u),apply(w,tlab,tcomplete,u),u]])
