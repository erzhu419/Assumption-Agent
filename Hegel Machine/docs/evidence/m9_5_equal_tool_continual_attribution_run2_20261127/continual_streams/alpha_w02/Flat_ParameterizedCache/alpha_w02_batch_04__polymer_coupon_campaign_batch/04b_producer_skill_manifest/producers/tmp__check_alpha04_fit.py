import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('s','/holdout_task/alpha_w02_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open('/holdout_task/public_alpha_w02_batch_04.pkl','rb'))[0]; d=p['dataset']; tr,te=d['train'],d['test']; dt,t,sy=d['past_dt'],d['eval_times'],d['yield_stress']; c=d['calibration_load_cells']; cal=s._calibrate(c,sy)
n=len(tr['lab_id']); feat=s._features(tr,te); xx=s._inputs(tr,feat[:n],cal,dt,sy); xt=s._inputs(te,feat[n:],cal,dt,sy)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; g=p['gt']; yt=np.c_[g['te_clean']-te['strain_0'][:,None],g['te_true_rate0']]
complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); complete_te=np.isfinite(te['coupon_features_raw']).all(axis=1)
print('training coverage',[(lab,int((tr['lab_id']==lab).sum()),int(((tr['lab_id']==lab)&complete).sum())) for lab in range(4)])
def rmse(e):
 return np.sqrt(np.nanmean(e**2))
def metrics(e):
 return np.round([rmse(e[:,:-1]),rmse(e[:,-1])],6).tolist()
def subset(x,m):
 return tuple(a[m] for a in x)
for name, eligible in [('all',np.ones(n,bool)),('complete',complete)]:
 pred=np.empty_like(y)
 for lab in range(4):
  m=tr['lab_id']==lab; fit=~m&eligible
  tau,w=s._fit(subset(xx,fit),dt,t,y[fit]); pred[m]=s._design(subset(xx,m),dt,t,tau)@w
 err=pred-y
 print(name,'LOLO by lab / complete',[(i,metrics(err[tr['lab_id']==i]),metrics(err[(tr['lab_id']==i)&complete])) for i in range(4)])
 tau,w=s._fit(subset(xx,eligible),dt,t,y[eligible]); err=s._design(xt,dt,t,tau)@w-yt
 print(name,'tau',tau,'coef',w,'public complete/missing',metrics(err[complete_te]),metrics(err[~complete_te]),'public by lab',[metrics(err[te['lab_id']==i]) for i in range(4)])
# Three spot checks can include one glitch; test whether historical agreement rejects spurious exact pairs.
for lab in range(4):
 ii=np.flatnonzero(c['lab_id']==lab); ii=ii[np.argsort(c['cmd_stress'][ii])]
 scenarios={}
 for inds in (ii[[0,len(ii)//2,-1]],ii[[3,4,5]]):
  for bad in inds:
   for glitch in (-.22,.22):
    keep=(c['lab_id']!=lab)|np.isin(np.arange(len(c['lab_id'])),inds)
    cc={k:v.copy() for k,v in c.items()}; cc['true_stress'][bad]+=glitch; cc={k:v[keep] for k,v in cc.items()}
    got=s._calibrate(cc,sy)
    x=np.r_[tr['past_stress_cmd'][tr['lab_id']==lab].ravel(), tr['stress_hold_cmd'][tr['lab_id']==lab]]
    delta=(got-cal)[lab]; err=np.max(abs(x*delta[0]+delta[1])); scenarios[(tuple(map(int,inds)),int(bad),glitch)]=err
 key=max(scenarios,key=scenarios.get)
 print('three-spot EMI',lab,'worst physical stress delta',round(scenarios[key],6),'case',key)
