import importlib.util, pickle
import numpy as np
from scipy.stats import spearmanr

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c02_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(root+'public_alpha_m97_c02_s00.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
cal = s._calibration(d)
p, h = s._physical(tr, cal)
pte, hte = s._physical(te, cal)
print('linear regime margins: banding', .50-d['shear_banding_index'], 'past stress train/test', [d['crit_stress']-np.abs(a).max() for a in (p,pte)])
n = len(p)
X = np.c_[np.ones(n), s._align(d)[:n]]
y = tr['strain_curves_obs']-tr['strain_0'][:,None]
v = tr['instant_rate_obs']
lab, stratum = tr['lab_id'], tr['stratum']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
r, q = np.empty_like(y), np.empty_like(v)
for l in np.unique(lab):
    held = lab == l
    u,w = s._fit(p[~held],h[~held],dt,t,X[~held],y[~held],v[~held])
    a,b = s._design(u,p[held],h[held],dt,t,X[held])
    r[held], q[held] = a@w-y[held], b@w-v[held]
row = np.median(r[:, -4:]/t[-4:], axis=1)
print('held-out complete coupons by lab: n, signed late slope, median absolute late slope, signed rate, median absolute rate')
for l in np.unique(lab):
    m = (lab == l) & complete
    print(l, m.sum(), np.round([np.median(row[m]),np.median(abs(row[m])),np.median(q[m]),np.median(abs(q[m]))],6))
print('held-out complete coupons by stratum: n, signed late slope and rate')
for j in np.unique(stratum):
    m = (stratum == j) & complete
    print(j, m.sum(), np.round([np.median(row[m]),np.median(q[m])],6))
for name,x in [('hold stress',h),('abs hold stress',abs(h)),('peak history stress',abs(p).max(1))]:
    def rho(m):
        return [spearmanr(x[m],z[m]).statistic for z in (row,q)]
    print(name, 'pooled then within lab late slope / rate rank correlations', np.round([rho(complete)]+[rho(complete & (lab == l)) for l in np.unique(lab)],3).tolist())
u = np.log(s.solve_alpha_m97_c02_s00(d)['estimated_taus'])
_,_,da,db = s._design(u,p,h,dt,t,X,True)
for j in range(2):
    step = np.eye(2)[j]*1e-5
    ap,bp = s._design(u+step,p,h,dt,t,X)
    am,bm = s._design(u-step,p,h,dt,t,X)
    np.testing.assert_allclose((ap-am)/(2e-5), np.eye(7)[j]*da[:,:,j,None], atol=1e-8)
    np.testing.assert_allclose((bp-bm)/(2e-5), np.eye(7)[j]*db[:,j,None], atol=1e-8)
print('Both branch Jacobians agree with finite differences on measured histories.')
