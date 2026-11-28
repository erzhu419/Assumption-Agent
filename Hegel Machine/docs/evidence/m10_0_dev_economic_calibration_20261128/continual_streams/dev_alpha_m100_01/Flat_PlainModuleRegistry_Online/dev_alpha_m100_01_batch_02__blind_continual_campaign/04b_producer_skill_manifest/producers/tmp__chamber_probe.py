import sys, json, pickle, importlib.util
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=p['dataset']; diagnostic=p['campaign_diagnostics']
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_alpha_m100_01_batch_01_solver.py'); prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
cal=calibrate_upstream(raw); train,test=cal['train'],cal['test']; dg=calibrate_upstream(dict(raw,test=diagnostic))['test']; n=len(train['lab_id']); xx=prior._features(train,test); dt,t=cal['past_dt'],cal['eval_times']; keys=('past_channel_raw','hold_channel_raw')
def loads(p): return tuple(p[k].sum(axis=-1) for k in keys)
def obs(p): return np.c_[p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']]
def diagfeat():
 def combine(k): return np.concatenate([train[k],test[k]])
 labs=combine('lab_id'); refs=combine('ref_coupons'); x=dg['coupon_features_raw'].copy()
 for lab in np.unique(labs):
  r=refs[labs==lab]; w=dg['lab_id']==lab; x[w]=(x[w]-np.median(r.mean(0)))/np.median(r.std(0))
 return x,np.where(np.isfinite(x),x,prior._features(train,test).mean(0))
def fit(idx,scale=(.008,.02)):
 past,hold=loads(train); y=obs(train)[idx]; sc=np.r_[[scale[0]]*len(t),scale[1]]
 def arr(v):
  c,r=prior._design(np.exp(v),past[idx],hold[idx],xx[:n][idx],dt,t); return np.concatenate((c,r[:,None,:]),axis=1)
 def fun(b): return ((arr(b[:2])@b[2:]-y)/sc).ravel()
 v=np.log([.4,2.5]); a=arr(v); beta=np.linalg.lstsq((a/sc[None,:,None]).reshape(-1,7),(y/sc).ravel(),rcond=None)[0]; q=least_squares(fun,np.r_[v,beta],bounds=(np.r_[np.log([.05001,.05001]),[-np.inf]*7],np.r_[np.log([24.999,24.999]),[np.inf]*7]),loss='cauchy',max_nfev=120)
 return np.exp(q.x[:2]),q.x[2:]
def basis(ta): return np.c_[np.vstack((-np.expm1(-t[:,None]/ta),1/ta)),np.r_[t,1]]
if __name__=='__main__':
 xraw,x=diagfeat(); past,hold=loads(dg); y=obs(dg); ta,th=fit(np.arange(n)); print('material',ta,th); c,r=prior._design(ta,*loads(train),xx[:n],dt,t); print('train abs error quantiles',np.quantile(abs(obs(train)-np.c_[c@th,r@th]),[.5,.9,.95,.99],axis=0).round(4)); c,r=prior._design(ta,past,hold,x,dt,t); base=np.c_[c@th,r@th]; ac,ar=prior._chamber(ta,th,hold,x,t); arch=np.c_[ac,ar]; b=basis(ta); q=y@np.linalg.pinv(b.T); print('baseline rmse',np.mean((y-base)**2,axis=0)**.5,'archive rmse',np.mean((y-arch)**2,axis=0)**.5); print('free basis residual maxima per coupon',abs(y-q@b.T).max(1).round(4)); print('condition',np.linalg.cond(b)); print('diagnostic variation\n',np.c_[dg['lab_id'],dg['stratum'],xraw,hold,past,q].round(4)); print('calibration payload differences',*[np.max(np.abs(cal[k][a]-calibrate_upstream(raw,json.load(open('/holdout_task/_upstream_param_payload.json')))[k][a])) for k in ['train','test'] for a in keys]);
