import json
import pickle
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import batch_solver
import downstream_solver as s
from _verified_upstream import calibrate_upstream
root = '/holdout_task/'
with open(root+'public_batch.pkl','rb') as f:
    item = pickle.load(f)[0]
with open(root+'_upstream_param_payload.json') as f:
    p = json.load(f)
raw = item['dataset']
d = calibrate_upstream(raw,p)
q = calibrate_upstream(dict(raw,test=item['campaign_diagnostics']),p)['test']
a = batch_solver.solve_batch(raw)
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
x, xt, xq = s._features(tr,te,q)
taus, b = a['estimated_taus'], a['hereditary_weights']
assert .05 < taus[0] < taus[1] < 25 and all(np.isfinite(v).all() for v in a.values())
for k, v in s.solve_downstream(d).items():
    np.testing.assert_array_equal(v,a[k])
base = s._design(q,xq,dt,t,taus) @ b
f = s._chamber_basis(q,xq,dt,t,taus,b)
y = s._response(q)
for m in [te, q]:
    xx = xt if m is te else xq
    ts = np.array([0., 1.e-6])
    r = s._design(m,xx,dt,ts,taus) @ b + s._chamber_basis(m,xx,dt,ts,taus,b) @ s._CHAMBER
    np.testing.assert_allclose((r[:,1]-r[:,0])/ts[1],r[:,-1],atol=2.e-5,rtol=2.e-6)

def fit(mask):
    z = f[mask].reshape(-1,f.shape[-1]); target = (y-base)[mask].ravel()
    start = np.linalg.lstsq(z,target,rcond=None)[0]
    return least_squares(lambda w: z@w-target,start,loss='soft_l1',f_scale=.014).x

np.testing.assert_allclose(fit(np.ones(len(y),bool)),s._CHAMBER,atol=.00011,rtol=0)
lab = q['lab_id']
worst = np.zeros(4)
for j in np.unique(lab):
    m = lab==j
    e = np.abs((y-base-f@fit(~m))[m]); scores = np.array([e[:,:3].mean(),e[:,-4:-1].mean(),e[:,-2].mean(),e[:,-1].mean()]); worst = np.maximum(worst,scores)
    print('held-out lab',j,'early, late, endpoint, rate MAE',np.round(scores,5))
assert np.all(worst < [.018,.024,.024,.028]), worst
print('worst',np.round(worst,5),'taus',taus,'theta',b)
