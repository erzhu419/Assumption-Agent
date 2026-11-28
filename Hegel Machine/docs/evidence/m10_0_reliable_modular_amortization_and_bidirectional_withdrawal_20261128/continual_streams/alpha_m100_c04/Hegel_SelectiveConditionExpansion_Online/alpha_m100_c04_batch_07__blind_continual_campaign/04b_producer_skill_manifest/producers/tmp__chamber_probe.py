import sys, json, pickle, importlib.util
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
base='/tmp/hegel_continual_archive/'
def load(name):
 s=importlib.util.spec_from_file_location(name,base+name+'_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m=load('alpha_m100_c04_batch_04'); a=load('alpha_m100_c04_batch_01')
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=it['dataset']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); ds=calibrate_upstream(raw,pay); dg=calibrate_upstream(dict(raw,test=it['campaign_diagnostics']),pay)['test']; tr,te=ds['train'],ds['test']; dt,t=ds['past_dt'],ds['eval_times']; n=len(tr['lab_id']); x=m._features(tr,te)
def obs(p): return p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']
taus,w=m._fit(m._inputs(tr,x[:n]),dt,t,*obs(tr)); print('fit',taus,w)
def metric(y,r,p,q):
 for v in (p-y,q-r): print('rmse/mae/median/max',np.round([np.sqrt(np.mean(v*v)),np.mean(abs(v)),np.median(abs(v)),np.max(abs(v))],5),end=' ')
 print()
for k,p in [('train',tr),('diagnostic',dg)]:
 xx=x[:n] if k=='train' else m._features(tr,dg)[n:]; inp=m._inputs(p,xx); past,hold,c=inp; y,r=obs(p); A,B=m._design(taus,inp,dt,t); print(k,'baseline');metric(y,r,A@w,B@w)
 if k=='train':continue
 for name,f in [('batch01',a._chamber_response),('batch04',m._chamber_response)]:
  print(name);metric(y,r,*f(taus,w,hold,c,t))
 rise=-np.expm1(-t[:,None]/taus); design=np.vstack([np.column_stack([rise,t]),np.r_[1/taus,1]]); target=np.column_stack([y,r]); coef=np.linalg.lstsq(design,target.T,rcond=None)[0].T
 d=c@w[2:]; gap=B[:,:2]*taus; f=np.column_stack([np.ones(len(c)),hold,d,gap,past,c[:,1:]])
 print('coef residual'); metric(y,r,coef@design[:-1].T,coef@design[-1]); print('correlation with [1,hold,drift,gap0,gap1,past0:6,x0:4]',np.round(np.corrcoef(np.column_stack([coef,f[:,1:]]).T)[:3,3:],2)); print('amplitude/drift/cov',np.round(np.column_stack([coef,hold,d,gap]),4))
