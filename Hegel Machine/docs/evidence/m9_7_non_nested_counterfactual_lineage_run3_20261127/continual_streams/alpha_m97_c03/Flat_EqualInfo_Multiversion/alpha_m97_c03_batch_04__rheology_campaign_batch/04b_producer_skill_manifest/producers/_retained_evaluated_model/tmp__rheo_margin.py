import importlib.util,pickle,numpy as np
P='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',P+'alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(P+'public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; lab=tr['lab_id']; st=tr['stratum']; n=len(lab)
p,h,x,o,dt,t=m._prepare(d); D=m._damage(d,p,dt)[:n]; p,h,x=p[:n],h[:n],x[:n]
y=np.column_stack((tr['strain_curves_obs']-o[:n,None], tr['instant_rate_obs']))
def args(i): return p[i],h[i],x[i],dt,t,D[i]
def metrics(e):
 return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.median(np.abs(e[:,:-1])),np.median(np.abs(e[:,-1])),np.max(np.abs(e[:,:-1])),np.max(np.abs(e[:,-1]))])
import json
from unittest.mock import patch

# Look for transfer failures specific to intersections or to missing coupons.
e = np.empty_like(y)
for c, s in np.unique(np.column_stack((lab, st)), axis=0):
    v = (lab == c) & (st == s)
    pp, w = m._fit((*args(~v), True), y[~v])
    e[v] = m._basis(pp, *args(v), True) @ w - y[v]
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
for miss in (False, True):
    for active in (False, True):
        v = (missing == miss) & ((D > 0) == active)
        a = e[v]
        if len(a):
            print('missing/active/n', miss, active, len(a), 'max horizon median abs', round(np.median(np.abs(a[:, :-1]), axis=0).max(), 6), 'rate median abs', round(np.median(np.abs(a[:, -1])), 6), 'signed medians', np.round(np.median(a, axis=0), 5).tolist())

# Use prior sweeps separately rather than assuming their differences are negligible.
reference = m.solve_alpha_m97_c03_batch_04(d)
with open(m._ARCHIVE) as f:
    history = json.load(f)
for sweep in history[:-1]:
    def prior(c):
        v = np.asarray(sweep['lab_id']) == c
        return np.polyfit(np.asarray(sweep['cmd_stress'])[v], np.asarray(sweep['true_stress'])[v], 1)
    with patch.object(m, '_prior_calibration', side_effect=prior):
        other = m.solve_alpha_m97_c03_batch_04(d)
    print('prefix sweep', sweep['episode_short_name'], 'max absolute changes', {k: round(float(np.max(np.abs(np.asarray(other[k]) - reference[k]))), 7) for k in reference})
