import importlib.util,pickle,numpy as np
P='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',P+'alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(P+'public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; lab=tr['lab_id']; st=tr['stratum']; n=len(lab)
p,h,x,o,dt,t=m._prepare(d); D=m._damage(d,p,dt)[:n]; p,h,x=p[:n],h[:n],x[:n]
y=np.column_stack((tr['strain_curves_obs']-o[:n,None], tr['instant_rate_obs']))
def args(i): return p[i],h[i],x[i],dt,t,D[i]
def metrics(e):
 return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.median(np.abs(e[:,:-1])),np.median(np.abs(e[:,-1])),np.max(np.abs(e[:,:-1])),np.max(np.abs(e[:,-1]))])
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
phi = np.r_[np.log1p(t), 1.]
results = [np.empty_like(y), np.empty_like(y)]
for c, s in np.unique(np.column_stack((lab, st)), axis=0):
    v = (lab == c) & (st == s)
    for j, use in enumerate((~v, ~v & ~missing)):
        pp, ww = m._fit((*args(use), True), y[use])
        results[j][v] = y[v] - m._basis(pp, *args(v), True) @ ww
for j, e in enumerate(results):
    print('fit', ['all coupons','complete coupons only'][j])
    worst = np.zeros(2)
    for miss in (False, True):
        for active in (False, True):
            v = (missing == miss) & ((D > 0) == active)
            a = e[v]
            # If incorrect imputation is the only systematic error, residuals lie in span(phi), including the initial rate.
            beta = np.median(a[:, :-1] / phi[:-1], axis=1)
            transverse = a - beta[:, None] * phi
            raw = np.r_[np.median(np.abs(a[:, :-1]), axis=0).max(), np.median(np.abs(a[:, -1]))]
            off = np.r_[np.median(np.abs(transverse[:, :-1]), axis=0).max(), np.median(np.abs(transverse[:, -1]))]
            print('missing/active', miss, active, 'raw', raw, 'off-log', off, 'median abs drift shift', np.median(np.abs(beta)))
            worst = np.maximum(worst, raw)
    print('worst slice', worst)
for use in (np.ones(n, dtype=bool), ~missing):
    pp, ww = m._fit((*args(use), True), y[use]); print('parameters', np.exp(pp), ww)
