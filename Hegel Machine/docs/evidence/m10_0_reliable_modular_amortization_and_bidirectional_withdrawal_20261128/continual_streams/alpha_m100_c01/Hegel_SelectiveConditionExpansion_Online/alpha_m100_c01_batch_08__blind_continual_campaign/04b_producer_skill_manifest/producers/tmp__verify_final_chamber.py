import sys,runpy,copy,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
sys.argv=['/holdout_task/validate_public_schema.py','/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0],run_name='__main__')
from batch_solver import solve_batch
from downstream_solver import solve_downstream,_features,_predict,_design
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
tr,te,q=(d[k] for k in ('train','test','chamber_verification_trials'))
a=solve_batch(copy.deepcopy(item['dataset'])); b=solve_downstream(d)
for k in a:
    np.testing.assert_allclose(a[k],b[k],atol=1e-10,rtol=1e-10)
ts,w=a['estimated_taus'],a['hereditary_weights']; assert .05<ts[0]<ts[1]<25
_,x=_features(q,q); t=d['eval_times']; dt=d['past_dt']; h=q['hold_channel_raw'].sum(-1); p=q['past_channel_raw'].sum(-1)
curve,rate=_predict(q,x,ts,w,t); short=_predict(q,x,ts,w,np.array([0.,1e-6]))[0]
np.testing.assert_allclose(short[:,0],q['strain_0'],atol=1e-12)
np.testing.assert_allclose((short[:,1]-short[:,0])*1e6,rate,atol=2e-5,rtol=1e-6)
c,r=_design(ts,p,h,np.c_[np.ones(len(x)),x],dt,t)
e=c@w+q['strain_0'][:,None]-q['strain_curves_obs']; er=r@w-q['instant_rate_obs']; loss=curve-q['strain_curves_obs']; lossr=rate-q['instant_rate_obs']; labs=q['lab_id']; missing=np.isnan(q['coupon_features_raw']).any(1)
rms=lambda a:np.sqrt(np.mean(a*a))
for label,sel in [('complete',~missing),('missing',missing),('low hold',h<=np.median(h)),('high hold',h>np.median(h)),*[(f'lab {l}',labs==l) for l in np.unique(labs)]]:
    print(label,'deployed strain / rate RMSE',round(rms(loss[sel]),6),round(rms(lossr[sel]),6),'retained history rate RMSE',round(rms(er[sel]),6))
    assert rms(loss[sel])<.05 and rms(lossr[sel])<.04
print('retained vs reset rate P90:',np.quantile(abs(er),.9),np.quantile(abs(lossr),.9))
assert np.quantile(abs(er),.9)>10*np.quantile(abs(lossr),.9)
print('Verified calibrated composition, reset boundary and time-zero derivative.')
