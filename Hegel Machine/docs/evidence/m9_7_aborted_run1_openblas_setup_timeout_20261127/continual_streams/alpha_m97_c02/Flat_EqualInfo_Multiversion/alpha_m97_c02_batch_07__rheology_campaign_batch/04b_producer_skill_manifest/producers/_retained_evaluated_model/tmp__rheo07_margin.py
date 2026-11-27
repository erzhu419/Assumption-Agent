import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c02_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c02_batch_07.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']; lab = tr['lab_id']; n = len(lab)
a, z0, dt, times, plateau = s._prepare(d)
a = tuple(v[:n] for v in a)
y = tr['strain_curves_obs'] - z0[:n, None]; r = tr['instant_rate_obs']
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
# Fixed alternatives for an omitted terminal-drift interaction, all with
# the same derivative at zero as their curve amplitude.
x = a[2][:, 1:]; hold, act = a[1], a[3]
q = np.c_[x[:,0]*x[:,1], x[:,2]*x[:,3], x[:,0]*x[:,1]*x[:,2],
          (x[:,0]-x[:,1])/(1+(x[:,2]+x[:,3])**2),
          act*hold/(1+hold**2)]
phi = np.log1p(times) if plateau else times

def augment(c, v, ids):
    return np.concatenate((c, q[ids,None,:]*phi[None,:,None]),axis=2), np.c_[v,q[ids]]

for mode in ('baseline', 'complete-fit', 'drift-challenger'):
    es, er, params = np.empty_like(y), np.empty_like(r), []
    for ch in np.unique(lab):
        u = np.flatnonzero((lab != ch) & (~missing if mode == 'complete-fit' else True))
        v = np.flatnonzero(lab == ch)
        p,w = s._fit(tuple(z[u] for z in a),dt,times,plateau,y[u],r[u])
        params.append(np.r_[np.exp(p),w])
        cv,rv = s._design(p,tuple(z[v] for z in a),dt,times,plateau)
        if mode == 'drift-challenger':
            # Profile only the extra drift coefficients, so the audit asks
            # whether they improve an otherwise identical constitutive fit.
            cu,ru = s._design(p,tuple(z[u] for z in a),dt,times,plateau)
            from scipy.optimize import least_squares
            ec,ee = cu@w-y[u],ru@w-r[u]
            sc,sr = [max(1e-6, 1.4826*np.median(abs(e-np.median(e)))) for e in (ec,ee)]
            bu,du = augment(cu,ru,u); cv,rv = augment(cv,rv,v)
            def residual(k):
                return np.r_[((bu[:,:,9:]@k+ec)/sc).ravel(), (du[:,9:]@k+ee)/sr]
            w = np.r_[w,least_squares(residual,np.zeros(q.shape[1]),loss='cauchy',f_scale=2.5).x]
        es[v] = abs(cv@w-y[v]); er[v] = abs(rv@w-r[v])
    def worst(subsets):
        return np.round(np.max([np.r_[np.median(es[m]),np.median(er[m]),np.median(es[m,-3:])] for m in subsets if m.any()], axis=0),5)
    print(mode, 'worst curve/rate/late medians:',
          'lab x damage',worst([(lab==i)&((act>0)==j) for i in np.unique(lab) for j in (0,1)]),
          'imputed lab',worst([(lab==i)&missing for i in np.unique(lab)]),
          'complete lab',worst([(lab==i)&~missing for i in np.unique(lab)]),
          'tau span',np.round(np.ptp(params,axis=0)[:2],5))
