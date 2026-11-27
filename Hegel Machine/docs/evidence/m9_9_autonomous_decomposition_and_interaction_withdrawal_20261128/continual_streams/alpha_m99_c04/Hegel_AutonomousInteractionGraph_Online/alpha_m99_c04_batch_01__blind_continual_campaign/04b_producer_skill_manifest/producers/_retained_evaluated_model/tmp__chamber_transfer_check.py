import pickle,runpy,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/holdout_task/batch_solver.py'); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; tr,te,d=ds['train'],ds['test'],item['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']; s=m['_stress_map'](ds['calibration_load_cells']); xt,xe=m['_features'](tr,te); q,b=m['_fit'](tr,xt,s,dt,t)
def empty(g): return {k:v[:0] for k,v in g.items()}
x=m['_features'](d,empty(d))[0]; xl=m['_features'](te,empty(te))[0]; h=s[0]+d['hold_channel_raw']@s[1:]; y=np.c_[d['strain_curves_obs']-d['strain_0'][:,None],d['instant_rate_obs']]; labs=d['lab_id']; miss=np.isnan(d['coupon_features_raw']); r=b[2]+x@b[3:]; w=np.r_[np.ones(len(t)),2.5]; base=h*(b[:2]/q).sum()+r; print('deployment joint versus local drift RMS',np.sqrt(np.mean(((xe-xl)@b[3:])**2))); print('initial-rate conservation residual complete/missing',*[np.sqrt(np.mean((base-y[:,-1])[z]**2)) for z in [~miss.any(1),miss.any(1)]]); print('missing drift adjustments observed vs zero-fill',np.c_[np.flatnonzero(miss.any(1)),(y[:,-1]-base)[miss.any(1)],((-x*miss)@b[3:])[miss.any(1)]].round(6))
def model(p,dr,physical):
 q0,q1=np.exp(p[:2]); poles=np.array([q0,q1]); E=np.r_[-np.expm1(-t[:,None]/poles),[1/poles]]
 if physical:
  a0=h*p[2]+dr*(q0*p[3]); a1=h*((b[:2]/q).sum()-p[2]/q0)*q1+dr*(q1*(1-p[3])); A=np.c_[a0,a1]
 else:
  A=h[:,None]*p[2:4]+dr[:,None]*p[4:6]
 return A@E.T
def rmse(e): return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))])
for physical in (False,True):
 for mode in ('all','complete','zero'):
  dr=r if mode!='zero' else r-(x*miss)@b[3:]; pred=np.empty_like(y); par=[]
  for lab in np.unique(labs):
   test=labs==lab; fit=~test & (~miss.any(1) if mode=='complete' else True); start=np.r_[np.log(q),.64,.455] if physical else np.r_[np.log(q),.64,.44,.255,2.07]; opt=least_squares(lambda p:((model(p,dr,physical)-y)[fit]*w).ravel(),start); pred[test]=model(opt.x,dr,physical)[test]; par.append(opt.x)
  err=pred-y; print('conserved' if physical else 'free',mode,'pooled',rmse(err),'worst lab',np.max([rmse(err[labs==l]) for l in np.unique(labs)],axis=0),'missing',rmse(err[miss.any(1)]),'poles',np.exp(np.mean(par,axis=0)[:2]))
