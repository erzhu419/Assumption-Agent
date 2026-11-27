import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import batch_solver as b, downstream_solver as s
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d = b._up.calibrate_upstream(item['dataset'])
a, te = d['train'], d['test']; x, _ = s._features(a, te)
dt, t = d['past_dt'], d['eval_times']; bridge = np.r_[0., np.ones(4)]
c = item['dataset']['calibration_load_cells']; truth = c['true_channel_loads']; raw = c['bridge_raw']; order = np.random.default_rng(17).permutation(len(raw))
for i, va in enumerate(np.array_split(order, 4)):
    tr = np.setdiff1d(order, va)
    model = b._up._fit(truth[tr], raw[tr]); recovered = b._up._invert(raw[va], model) if hasattr(b._up,'_invert') else None
    if recovered is not None:
        e = recovered-truth[va]; print('bridge fold',i,'channel p90/max',np.quantile(abs(e),.9),np.max(abs(e)),'total stress p90',np.quantile(abs(e.sum(-1)),.9))
for lab in np.unique(a['lab_id']):
    mask = a['lab_id'] != lab
    tr = {k:v[mask] for k,v in a.items()}; va = {k:v[~mask] for k,v in a.items()}
    taus, theta = s._material(tr,bridge,x[mask],dt,t)
    def residual(q, sp, sx):
        cv, rv = s._basis(np.exp(q[:2]),*s._loads(sp,bridge),sx,dt,t)
        return cv@q[2:] + sp['strain_0'][:,None]-sp['strain_curves_obs'],rv@q[2:]-sp['instant_rate_obs']
    q = np.r_[np.log(taus),theta]
    er = residual(q,tr,x[mask]); scales = [max(.0005,1.4826*np.median(abs(e-np.median(e)))) for e in er]
    def obj(p): return np.concatenate([(e/z).ravel() for e,z in zip(residual(p,tr,x[mask]),scales)])
    bounds = (np.r_[np.log([.050001]*2),[-np.inf]*7],np.r_[np.log([24.99999]*2),[np.inf]*7])
    challenger = least_squares(obj,q,loss='cauchy',f_scale=2.5,bounds=bounds).x
    for name,p in [('soft_l1',q),('cauchy',challenger)]:
        e,r = residual(p,va,x[~mask]); print('lab',lab,name,'taus',np.exp(p[:2]).round(4),'p90 strain/rate',np.round([np.quantile(abs(e),.9),np.quantile(abs(r),.9)],5),'signed medians',np.round([np.median(e),np.median(r)],5),'early/late p90',np.round([np.quantile(abs(e[:,:3]),.9),np.quantile(abs(e[:,-3:]),.9)],5))
print('chamber labels available:',item.get('campaign_diagnostics') is not None)
