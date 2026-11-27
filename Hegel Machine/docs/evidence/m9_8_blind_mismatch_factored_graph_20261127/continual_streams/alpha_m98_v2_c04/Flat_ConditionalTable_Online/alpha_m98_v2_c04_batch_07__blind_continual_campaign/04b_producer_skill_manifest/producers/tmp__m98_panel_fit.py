import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,q=d['train'],d['test'],i['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; b=m._bridge(d['calibration_load_cells']); x=m._features(tr,te); n=len(tr['lab_id']); tau,th=m._fit(x[:n],*m._loads(tr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); past,h=m._loads(q,b)
keys=('lab_id','stratum','coupon_features_raw','ref_coupons'); a={k:np.concatenate((tr[k],te[k])) for k in keys}; emp={k:q[k][:0] for k in keys}; X=m._features(emp,q); Y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; g=q['lab_id']; bad=np.isnan(q['coupon_features_raw']).any(1); later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; G=(h[:,None]-past@(-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau)))*th[:2]; v=th[2]+X@th[3:]; strata=q['stratum']
for l in np.unique(g):
 def panel(p):
  p=p['ref_coupons'][p['lab_id']==l]; return np.r_[np.median(np.nanmean(p,axis=0)),np.median(np.nanstd(p,axis=0))]
 print('lab',l,'train/test panel',panel(a),'verification panel',panel(q))
def pred(p,mode):
 fast=0. if mode in ('fixed','fix_time','fix_drift') else p[0]; T=tau*np.exp([fast,p[1]]); S=np.exp(p[3]); D=np.exp(p[2])*v
 if mode=='fix_time': S=S*np.exp(p[4]*(h-np.median(h)))[:,None]
 if mode=='fix_drift': D=D*np.exp(p[4]*(h-np.median(h)))
 return np.c_[G@(-np.expm1(-t[None,:]/T[:,None]))-D[:,None]*S*np.expm1(-t/S),G@(1/T)+D]
def fit(mask,mode):
 return least_squares(lambda p:((pred(p,mode)-Y)[mask]/np.r_[np.full(len(t),.013),.024]).ravel(),[0.,.15,.21,.06]+([0.] if mode in ('fix_time','fix_drift') else []),loss='soft_l1',f_scale=1.5).x
def metrics(P):
 e=abs(P-Y); return np.array([*np.max([[np.median(e[g==l,:-1]),np.median(e[g==l,-1])] for l in np.unique(g)],axis=0),*np.max([[np.median(e[strata==l,:-1]),np.median(e[strata==l,-1])] for l in np.unique(strata)],axis=0),np.max(np.median(e[:,:-1],axis=0)),np.quantile(e[:,-2],.9),np.median(e[~bad,:-1]),np.median(e[bad,:-1])]).round(5)
old=np.r_[m._CHAMBER_LOG_TIME_SCALE,np.log(m._CHAMBER_DRIFT_GAIN),np.log(m._CHAMBER_DRIFT_TIME)]; print('deployed parameters on verification-specific panel',metrics(pred(old,'free')))
for mode in ('fixed','free','fix_time','fix_drift'):
 p=fit(np.ones(len(g),bool),mode); print(mode,'parameters',p.round(6),'fit',metrics(pred(p,mode)))
 for grp in (g,strata):
  P=np.zeros_like(Y)
  for l in np.unique(grp):
   a=grp==l; P[a]=pred(fit(~a,mode),mode)[a]
  print('LOO',metrics(P))
