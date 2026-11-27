import sys,json,pickle,importlib.util
sys.path.insert(0,'/holdout_task')
import numpy as np
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c01_batch_01_solver.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=i['dataset']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); d=calibrate_upstream(raw,pay); diag=calibrate_upstream({**raw,'test':i['campaign_diagnostics']},pay)['test']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']
def loads(s):return s['past_channel_raw'].sum(-1),s['hold_channel_raw'].sum(-1)
def obs(s):return s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']
x,xt=old._features(tr,te); _,xd=old._features(tr,diag)
tau,b=old._fit(x,*loads(tr),dt,t,*obs(tr));print('tau,b',tau,b)
for j in np.unique(tr['lab_id']):
 m=tr['lab_id']!=j; p,h=loads(tr); y,r=obs(tr); a,c=old._fit(x[m],p[m],h[m],dt,t,y[m],r[m]); f,g=old._design(a,x[~m],p[~m],h[~m],dt,t); print('LOLO',j,a,c,'med errors',np.median(abs(f@c-y[~m])),np.median(abs(g@c-r[~m])))
p,h=loads(diag);y,r=obs(diag);dr=b[2]+xd@b[3:]; a,c=old._design(tau,xd,p,h,dt,t); base=a@b; rr=c@b; prev,pr=old._chamber(tau,b,xd,h,t)
for label,q,s in [('baseline',base,rr),('prior chamber',prev,pr)]:print(label,'errors',np.sqrt(np.mean((y-q)**2)),np.sqrt(np.mean((r-s)**2)),np.median(abs(y-q)),np.median(abs(r-s)))
np.set_printoptions(precision=4,suppress=True,linewidth=160); print('diag residuals [hold,drift,rateobs,rateerr, curves at four times]'); print(np.c_[h,dr,r,r-rr,(y-base)[:,[0,2,6,11]]]);
from scipy.optimize import least_squares
# Fit diagnostic modal coefficients per coupon to inspect how the chamber alters memory.
f=np.c_[-np.expm1(-t[:,None]/tau),t]; cf=np.vstack((f,np.r_[1/tau,1])); modes=np.linalg.lstsq(cf,np.c_[y,r].T,rcond=None)[0].T; print('fixed-tau max specimen errors',np.sort(np.max(abs(modes@cf.T-np.c_[y,r]),axis=1)));print('modes',modes)
z=h[:,None]-c[:,:2]*tau; cols=np.c_[np.ones(len(h)),h,p,dr,z]; names=['1','h',*['p'+str(j) for j in range(p.shape[1])],'dr','z0','z1']; coef=np.linalg.lstsq(cols[:,:9],modes,rcond=None)[0];print('mode linear regression',list(zip(names,coef))); print('mode deviations',np.std(cols[:,:9]@coef-modes,axis=0))
