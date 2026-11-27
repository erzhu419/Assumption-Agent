import json,pickle,sys,glob,importlib.util
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('old',glob.glob('/tmp/hegel_continual_archive/*batch_01_solver.py')[0]); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['dataset']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); d=calibrate_upstream(raw,pay); g=calibrate_upstream({**raw,'test':item['campaign_diagnostics']},pay)['test']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; n=len(tr['lab_id']); nt=len(te['lab_id']); xs=old._features(tr,te,g)
def coupon(s,x):return old._coupon(s,x,np.r_[np.ones(4),0.])
def obs(s):return s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']
tau,theta=old._fit(coupon(tr,old._features(tr,te)[:n]),dt,t,*obs(tr)); p,h,x=coupon(g,xs[n+nt:]); dr=x@theta[2:]; yc,yr=obs(g); y=np.column_stack((yc,yr)); print('TAU THETA',tau,theta)
def basis(taus):return np.vstack((np.column_stack((-np.expm1(-t[:,None]/taus),t)),np.r_[1/taus,1]))
def amps(taus,y=y):return np.linalg.lstsq(basis(taus),y.T,rcond=None)[0].T
def err(l):
 b=basis(np.exp(l));return (amps(np.exp(l))@b.T-y).ravel()
bounds=(np.log([.05,.05]),np.log([50.,100.])); best=min([least_squares(err,np.log(s),bounds=bounds,loss='soft_l1',f_scale=.015,max_nfev=150) for s in [(0.12,1.1),(.4,2),(.7,4),(1.5,12)]],key=lambda z:z.cost); ct=np.sort(np.exp(best.x)); a=amps(ct); r=a@basis(ct).T-y
print('CHAMBER',ct,ct/tau,'resid',np.quantile(abs(r),[.5,.75,.9,.95,.99,1])); base=old._design(tau,(p,h,x),dt,t); print('BASE',np.sqrt(np.mean((base[0]@theta-yc)**2)),np.sqrt(np.mean((base[1]@theta-yr)**2)))
z=h[:,None]-base[1][:,:2]*tau
signals=np.column_stack((h,p,z,dr,x[:,1:],g['hold_channel_raw'],g['lab_id'],g['stratum'])); names=['h',*[f'p{i}' for i in range(6)],'z0','z1','dr','x0','x1','x2','x3',*[f'h{i}' for i in range(4)],'lab','stratum'];
for j in range(3):
 print('A',j,'range',np.quantile(a[:,j],[0,.25,.5,.75,1]),'corr',sorted(zip(names,np.round(np.corrcoef(a[:,j],signals.T)[0,1:],4)),key=lambda z:-abs(z[1]))[:12]);
print('specimen: hold, memory0, memory1, drift, fitted fast,slow,steady,max residual'); print(np.round(np.c_[h,z,dr,a,np.max(abs(r),axis=1)],4))
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('log',type(log),list(log) if isinstance(log,dict) else len(log)); entries=next(v for v in log.values() if isinstance(v,list)) if isinstance(log,dict) else log
for e in entries:
 print({k:(np.round(np.mean(v,axis=0),3).tolist() if k=='facility_telemetry_trace' else str(v)[:120]) for k,v in e.items()})
