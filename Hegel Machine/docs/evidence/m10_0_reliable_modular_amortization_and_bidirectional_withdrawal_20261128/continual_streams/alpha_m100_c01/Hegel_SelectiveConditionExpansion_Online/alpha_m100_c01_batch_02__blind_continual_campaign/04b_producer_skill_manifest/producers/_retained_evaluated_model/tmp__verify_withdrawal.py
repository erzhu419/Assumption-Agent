import sys,pickle,json
sys.path.insert(0,'/holdout_task')
import numpy as np
from batch_solver import solve_batch
from downstream_solver import solve_downstream,_features,_predict,_design
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
raw=item['dataset']; cal=calibrate_upstream(raw)
with open('/holdout_task/_upstream_param_payload.json') as f: payload=json.load(f)
q=calibrate_upstream({**raw,'test':item['campaign_diagnostics']},payload)['test']
out=solve_batch(raw); other=solve_downstream(cal)
for key in out: np.testing.assert_array_equal(out[key],other[key])
tau=out['estimated_taus']; theta=out['hereditary_weights']; assert .05<tau[0]<tau[1]<25
x=_features(q,q)[0]; d=cal['past_dt']; t=cal['eval_times']; y=q['strain_curves_obs']; r=q['instant_rate_obs']; labs=q['lab_id']; strata=q['stratum']; complete=np.isfinite(q['coupon_features_raw']).all(axis=1)
pred,rate=_predict(q,x,tau,theta,d,t); e=y-pred; er=r-rate
rms=lambda a: float(np.sqrt(np.mean(a*a)))
print('taus',tau,'weights',theta)
for name,labels in [('lab',labs),('stratum',strata),('missingness',complete),('coupon',np.arange(len(y)))]:
 print(name,'worst strain/rate',tuple(max(rms(a[labels==k]) for k in np.unique(labels)) for a in [e,er]))
print('worst horizon',max(rms(e[:,k]) for k in range(e.shape[1])))
print('pooled strain/rate',rms(e),rms(er))
c,r0=_design(tau,q['past_channel_raw'].sum(-1),q['hold_channel_raw'].sum(-1),np.c_[np.ones(len(x)),x],d,t)
print('unmodified benchtop chamber error',rms(y-q['strain_0'][:,None]-c@theta))
assert max(rms(e[labs==k]) for k in np.unique(labs)) < .015 and max(rms(er[labs==k]) for k in np.unique(labs)) < .012
small=np.array([0.,1.e-6]); p0,rr=_predict(q,x,tau,theta,d,small)
np.testing.assert_array_equal(p0[:,0],q['strain_0'])
np.testing.assert_allclose(np.diff(p0,axis=1).ravel()/small[1],rr,atol=1e-5,rtol=0)
print('zero-time strain and initial-rate derivative verified')
