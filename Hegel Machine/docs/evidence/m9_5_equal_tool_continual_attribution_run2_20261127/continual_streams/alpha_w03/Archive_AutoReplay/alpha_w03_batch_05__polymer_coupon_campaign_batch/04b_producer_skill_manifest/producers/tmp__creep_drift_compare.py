import importlib.util
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold

s = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_w03_batch_05_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
case = pickle.load(open('/holdout_task/public_alpha_w03_batch_05.pkl', 'rb'))[0]
d, g = case['dataset'], case['gt']
tr, te = d['train'], d['test']
x, h, p = m._prep(d)
dt, t, sy = (d[k] for k in ('past_dt', 'eval_times', 'yield_stress'))
labs = np.r_[tr['lab_id'], te['lab_id']]
raw = np.concatenate([tr['coupon_features_raw'], te['coupon_features_raw']])
missing = ~np.isfinite(raw)
n = len(tr['lab_id'])
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
z = np.r_[tr['instant_rate_obs'], te['instant_rate_obs']]
complete = ~missing.any(1)
rmse = lambda e: np.sqrt(np.mean(e**2))

def predict(train, val):
    train = train[complete[train]]
    tau, b = m._fit(x[train], h[train], p[train], dt, t, sy, y[train], z[train])
    c, r = m._design(tau, x, h, p, dt, t, sy)
    c, r = c @ b, r @ b
    late = t >= np.median(t)
    projected = np.median((y[train] - c[train])[:, late] / t[late], axis=1)
    drift = np.array([np.median(projected[labs[train] == lab]) for lab in range(4)])[labs[val]]
    innovation = z[val] - r[val]
    rres = z[train] - r[train]
    noise2 = (1.4826 * np.median(abs(rres-np.median(rres))))**2
    vprior = np.zeros(len(val))
    for j, row in enumerate(val):
        mask = missing[row]
        if mask.any():
            local = train[labs[train] == labs[row]]
            vprior[j] = np.var(x[local, 1:5][:, mask] @ b[3:7][mask], ddof=1)
    shrink = vprior / (vprior + noise2 + 1e-20)
    known = ~complete[val] & np.isfinite(innovation)
    zero = np.where(known, innovation, 0.)
    corr = np.stack([zero, np.where(known, innovation, drift), np.where(known, drift+shrink*(innovation-drift), drift)])
    return c[val][None, :, :] + corr[:, :, None]*t, r[val][None, :] + corr

Y = np.zeros((3,n,len(t)))
R = np.zeros((3,n))
for train,val in StratifiedKFold(4,shuffle=True,random_state=29).split(labs[:n], labs[:n]*10+tr['stratum']):
    Y[:,val], R[:,val] = predict(train,val)
E = Y-y
inlier = abs(E[0]) < .2
for label, sub in [('complete', complete[:n]), ('missing', ~complete[:n])]:
    groups = [(labs[:n]==l)&(tr['stratum']==j)&sub for l in range(4) for j in np.unique(tr['stratum'])]
    groups = [q for q in groups if q.sum() >= 3]
    print('OOF', label, 'pooled strain', [rmse(v[inlier&sub[:,None]]) for v in E], 'worst cohort strain', [max(rmse(v[inlier&q[:,None]]) for q in groups) for v in E])
    if label == 'complete':
        print('OOF worst complete cohort rate', [max(rmse((v-z[:n])[q]) for q in groups) for v in R])
Y, R = predict(np.arange(n),np.arange(n,len(labs)))
E = Y+te['strain_0'][None,:,None]-g['te_clean']
R = R-g['te_true_rate0']
for l in range(4):
    for miss in (False, True):
        q = (te['lab_id']==l)&(complete[n:] != miss)
        if q.any():
            print('public',l,miss,q.sum(),'strain',np.round([rmse(v[q]) for v in E],6),'rate',np.round([rmse(v[q]) for v in R],6))
print('columns: contractual baseline, lab drift, lab drift plus missing-feature shrinkage')
