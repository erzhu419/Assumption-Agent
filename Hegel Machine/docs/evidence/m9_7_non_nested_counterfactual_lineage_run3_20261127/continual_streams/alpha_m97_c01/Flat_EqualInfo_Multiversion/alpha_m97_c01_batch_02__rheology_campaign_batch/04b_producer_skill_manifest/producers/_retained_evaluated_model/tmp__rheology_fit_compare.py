import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); dt,t,past,hold,x=m._prepare(d)
yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; lab=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)

def predict(p,q):
 c,r,*_=m._basis(p[:2],dt,t,past[q],hold[q],x[q]); return c@p[2:],r@p[2:]

def refine(p,q,loss,include_missing):
 if not include_missing: q=q&complete
 y=np.column_stack((yc[q],yr[q])); c,r=predict(p,np.flatnonzero(q)); e=y-np.column_stack((c,r)); scale=np.maximum(1.4826*np.median(abs(e-np.median(e,axis=0)),axis=0),1e-6)
 def res(v):
  c,r=predict(v,np.flatnonzero(q)); return ((np.column_stack((c,r))-y)/scale).ravel()
 z=least_squares(res,p,loss=loss,f_scale=2.5,x_scale='jac',max_nfev=60).x
 return z

configs=[('baseline',None,True),('cauchy/all','cauchy',True),('cauchy/complete','cauchy',False),('softl1/complete','soft_l1',False)]
cv={k:[] for k,_,_ in configs}
for held in np.unique(lab):
 q=lab!=held; p=m._fit(dt,t,past[:n][q],hold[:n][q],x[:n][q],yc[q],yr[q]); h=~q
 for name,loss,inc in configs:
  z=p if loss is None else refine(p,q,loss,inc); c,r=predict(z,np.flatnonzero(h)); e1=abs(c-yc[h]); e2=abs(r-yr[h]); cc=complete[h]; cv[name].append([np.median(e1[cc]),np.quantile(e1[cc],.90),np.median(e2[cc]),np.quantile(e2[cc],.90),np.median(e1[~cc]),np.median(e2[~cc])])
p=m._fit(dt,t,past[:n],hold[:n],x[:n],yc,yr); tc=np.isfinite(te['coupon_features_raw']).all(axis=1)
for name,loss,inc in configs:
 z=p if loss is None else refine(p,np.ones(n,dtype=bool),loss,inc); c,r=predict(z,np.arange(n,len(x))); e1=c+te['strain_0'][:,None]-g['te_clean']; e2=r-g['te_true_rate0']; rm=lambda v:np.sqrt(np.mean(v*v))
 print(name,'LOLO worst [complete strain med/P90, rate med/P90; missing strain/rate med]',np.max(cv[name],axis=0).round(5),'taus',np.exp(z[:2]).round(5),'hw max error',round(np.max(abs(z[2:]-g['theta'])),5),'public complete worst-lab rmse',[round(max(rm(e[(te['lab_id']==j)&tc]) for j in np.unique(lab)),5) for e in (e1,e2)],'public all worst-lab rmse',[round(max(rm(e[te['lab_id']==j]) for j in np.unique(lab)),5) for e in (e1,e2)])
