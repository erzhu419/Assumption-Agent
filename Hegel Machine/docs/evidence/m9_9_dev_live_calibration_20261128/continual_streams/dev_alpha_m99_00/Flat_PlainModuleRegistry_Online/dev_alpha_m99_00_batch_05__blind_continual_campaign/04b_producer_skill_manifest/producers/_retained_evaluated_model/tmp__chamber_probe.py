import sys,pickle
sys.path.insert(0,'/holdout_task')
sys.path.insert(0,'/tmp/hegel_continual_archive')
import numpy as np
from _verified_upstream import calibrate_upstream
from dev_alpha_m99_00_batch_02_solver import _features,_fit,_design
from dev_alpha_m99_00_batch_01_solver import _chamber
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=calibrate_upstream(dict(p['dataset'],chamber_verification_trials=p['campaign_diagnostics']))
a,b,q=(d[k] for k in ('train','test','chamber_verification_trials'))
dt,t=d['past_dt'],d['eval_times']; n=len(a['lab_id']); x=_features(a,b); xq=_features(a,b,q)[len(x):]
def coupon(s,x): return s['past_channel_raw'].sum(-1),s['hold_channel_raw'].sum(-1),np.column_stack((np.ones(len(x)),x))
def obs(s): return s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']
def metrics(name,pred,s):
 e,f=(v-u for v,u in zip(pred,obs(s))); print(name, 'rmse',np.sqrt(np.mean(e*e)), np.sqrt(np.mean(f*f)), 'medabs', np.median(np.abs(e)),np.median(np.abs(f)), 'horizon rmse',np.round(np.sqrt((e**2).mean(0)),4))
tau,th=_fit(coupon(a,x[:n]),dt,t,*obs(a)); print('parameters',tau,th)
pa,ha,dr=coupon(q,xq); c,r=_design(tau,(pa,ha,dr),dt,t); drift=dr@th[2:]
metrics('baseline',(c@th,r@th),q)
metrics('batch01',_chamber(tau,th,ha,drift,t),q)
metrics('drift exp1',(c[:,:,:2]@th[:2]+drift[:,None]*-np.expm1(-t),r@th),q)
c0,r0=_design(tau,(pa*0,ha,dr),dt,t)
metrics('reset only',(c0@th,r0@th),q)
metrics('reset and exp1',(c0[:,:,:2]@th[:2]+drift[:,None]*-np.expm1(-t),r0@th),q)
cy,ry=obs(q); A=np.column_stack((c0[:,:,0].ravel()*th[0],c0[:,:,1].ravel()*th[1],((c-c0)[:,:,:2]@th[:2]).ravel(),(drift[:,None]*t).ravel(),(drift[:,None]*-np.expm1(-t)).ravel())); print('exploratory pooled coefficients',np.linalg.lstsq(A,cy.ravel(),rcond=None)[0])
