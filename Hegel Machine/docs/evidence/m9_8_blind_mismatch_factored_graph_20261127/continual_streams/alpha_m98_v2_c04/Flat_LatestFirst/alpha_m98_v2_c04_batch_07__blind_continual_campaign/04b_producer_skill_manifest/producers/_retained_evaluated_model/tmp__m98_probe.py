import sys, pickle, importlib.util, itertools
import numpy as np
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=p['dataset']; q=p['campaign_diagnostics']; tr=ds['train']; te=ds['test']; t=ds['eval_times']; dt=ds['past_dt']; cal=ds['calibration_load_cells']
s=importlib.util.spec_from_file_location('arc','/tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_06_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
a,b=cal['true_channel_loads'],cal['bridge_raw']; print('cal load ranges',a.min(0),a.max(0),'basis sizes')
for name,Z in [('linear',np.c_[np.ones(len(a)),a]),('archive',m._bridge_basis(a)),('archive+sq',np.c_[m._bridge_basis(a),a*a]),('full cubic',np.c_[np.ones(len(a)),*[np.prod(a[:,idx],axis=1) for k in [1,2,3] for idx in itertools.combinations_with_replacement(range(4),k)]])]:
 pred=np.zeros_like(b); mag=np.max(abs(b-np.median(b,axis=0)),1); ini=mag<5*np.median(mag)
 for j in range(5):
  val=np.arange(len(a))%5==j; keep=ini & ~val
  for _ in range(5):
   C=np.linalg.lstsq(Z[keep],b[keep],rcond=None)[0]; err=np.linalg.norm(Z@C-b,axis=1); keep=(err<8*max(np.median(err[~val]),1e-8)) & ~val
  pred[val]=Z[val]@C
 print(name,Z.shape[1],np.quantile(np.linalg.norm(pred-b,axis=1),[.25,.5,.75,.85,.9,.95,1]))
B=m._fit_bridge(cal); print('archive coefs\n',np.round(B,5)); print('telemetry training range',np.quantile(tr['aux_chamber_telemetry'],[0,.5,1],axis=0)); print('telemetry test range',np.quantile(te['aux_chamber_telemetry'],[0,.5,1],axis=0)); print('q keys',list(q)); X=m._features(tr,te); xq=m._features(tr,q)[len(tr["lab_id"]):]; past,hold=m._loads(tr,B); pq,hq=m._loads(q,B); aq,rq=m._loads(te,B); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; yq=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; tau,th=m._fit(X[:len(past)],past,hold,dt,t,y,tr['instant_rate_obs']); print('train tau/theta',tau,th); C,R=m._design(X[:len(past)],past,hold,dt,t,tau); print('train errors',np.quantile(abs(C@th-y.ravel()),[.5,.9,.95,1]),np.quantile(abs(R@th-tr['instant_rate_obs']),[.5,.9,.95,1])); print('hold ranges',np.quantile(hq,[0,.5,1]),np.quantile(rq,[0,.5,1]));
def state(p,tau):
 z=np.zeros((len(p),2))
 for i,d in enumerate(dt):
  f=np.exp(-d/tau); z=f*z+(1-f)*p[:,i,None]
 return z
z=state(pq,tau); drift=th[2]+xq@th[3:]; Y=np.c_[yq,r];
def bases(tt): return np.vstack((np.c_[-np.expm1(-t[:,None]/tt),t],np.r_[1/tt,1]))
def fit(U):
 def pred(p): return U@p[2:].reshape(U.shape[1],3)@bases(np.exp(p[:2])).T
 p0=np.r_[np.log(tau),np.linalg.lstsq(U,Y@np.linalg.pinv(bases(tau).T),rcond=None)[0].ravel()]; oo=least_squares(lambda p:(pred(p)-Y).ravel(),p0,loss='cauchy',f_scale=.025,max_nfev=450); return np.exp(oo.x[:2]),oo.x[2:].reshape(U.shape[1],3),pred(oo.x)-Y
for name,U in [('reset',np.c_[np.ones(len(r)),hq,drift]),('memory',np.c_[np.ones(len(r)),hq,drift,z]),('free',np.eye(len(r)))]:
 tt,coef,err=fit(U); print(name,'tau',tt,'curve/rate abs',np.quantile(abs(err[:,:-1]),[.5,.9,1]),np.quantile(abs(err[:,-1]),[.5,.9,1])); print(np.round(coef,4))
 if name=='free': print('rows: lab hold drift z0 z1 amp0 amp1 rate',np.round(np.c_[q['lab_id'],hq,drift,z,coef],4))
