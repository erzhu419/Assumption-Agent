import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = item['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
xtr, xte = m._features(tr, te)
b = m._bridge(d['calibration_load_cells'])
s = m._prepare(tr, xtr, b)
s_te = m._prepare(te, xte, b)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
for label, part in [('train', s), ('deployment', s_te)]:
    print(label, 'hold range', np.quantile(part[1], [0, .1, .9, 1]).round(4), 'past range', np.quantile(part[0], [0, .1, .9, 1]).round(4))

def fit_alternative(f, tau, w, loss, rate_weight):
    ss = tuple(v[f] for v in s)
    A, B = m._design(np.log(tau), ss, dt, t)
    sc = max(m._mad(A@w-y[f]), 1e-7)
    sr = max(m._mad(B@w-r[f]), 1e-7)
    def residual(p):
        C,R = m._design(p[:2], ss, dt, t)
        return np.r_[((C@p[2:]-y[f])/sc).ravel(), rate_weight*(R@p[2:]-r[f])/sr]
    p = least_squares(residual, np.r_[np.log(tau),w], bounds=(np.r_[[np.log(.050001)]*2, [-np.inf]*7], np.r_[[np.log(24.9999)]*2,[np.inf]*7]), loss=loss, f_scale=2.5, max_nfev=120).x
    return np.exp(p[:2]), p[2:]

quantiles = np.quantile(s[1], [1/3, 2/3])
groups = [('lab', tr['lab_id']), ('stratum', tr['stratum']), ('hold', np.digitize(s[1], quantiles)), ('history', np.digitize(s[1]-s[0]@(dt/dt.sum()), np.quantile(s[1]-s[0]@(dt/dt.sum()), [1/3,2/3])))]
results = {}
for group, a in groups:
    for k in np.unique(a):
        h = a==k
        f = ~h
        tau,w = m._fit(tuple(v[f] for v in s), dt,t,y[f],r[f])
        fits = {'current': (tau,w), 'soft_l1': fit_alternative(f,tau,w,'soft_l1',1.), 'rate_upweight': fit_alternative(f,tau,w,'cauchy',1.6)}
        for name,(u,v) in fits.items():
            C,R = m._design(np.log(u), tuple(z[h] for z in s),dt,t)
            e,q = abs(C@v-y[h]), abs(R@v-r[h])
            results.setdefault(name,[]).append([np.quantile(e,.9),np.quantile(q,.9),np.quantile(e,.9,axis=0).max(),*np.sort(u)])
        print(group, k, 'baseline taus', np.round(tau,4), 'baseline p90 strain/rate', np.round(results['current'][-1][:2],5))
for name,rows in results.items():
    v = np.asarray(rows)
    print(name, 'worst p90 strain/rate/horizon',v[:,:3].max(0).round(5),'tau interval',v[:,3:].min(0).round(4),v[:,3:].max(0).round(4))
