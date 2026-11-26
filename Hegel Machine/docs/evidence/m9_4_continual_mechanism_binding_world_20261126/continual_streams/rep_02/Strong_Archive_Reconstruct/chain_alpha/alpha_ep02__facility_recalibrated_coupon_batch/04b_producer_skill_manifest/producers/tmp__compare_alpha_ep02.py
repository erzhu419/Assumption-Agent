import importlib.util, pickle
import numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_ep02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(p+'public_alpha_ep02.pkl','rb') as f: cases=pickle.load(f)
rmse=lambda a: np.sqrt(np.mean(a*a))
print('public cases',len(cases))
for i,r in enumerate(cases):
 d,g=r['dataset'],r['gt']; tr,te,y,ok=m._prepare(d); t=np.asarray(d['eval_times']); labs=np.asarray(d['train']['lab_id']); ids=np.unique(labs)
 # For the adaptive challenger, tune only on inner held-out labs.
 def innovations(excluded):
  xx=[]; zz=[]
  for inner in ids:
   if inner in excluded: continue
   fit=ok & ~np.isin(labs,(*excluded,inner)); q=ok & (labs==inner)
   ts,b=m._fit(tr,y,fit); pred=tr.matrix(np.log(ts))@b
   xx.extend((y[q,-1]-pred[q,-1]).tolist())
   k=t>=np.median(t); zz.extend(np.median((y[q,:-1]-pred[q,:-1])[:,k]/t[k],axis=1).tolist())
  return np.asarray(xx),np.asarray(zz)
 def learn(excluded):
  x,z=innovations(excluded); grid=np.linspace(0,1,21)
  return grid[np.argmin(np.median(np.abs(z[:,None]-x[:,None]*grid),axis=0))]
 def update(pred, rate, complete, w):
  return pred+((rate-pred[:,-1])*np.where(complete,w,1))[:,None]*np.r_[t,1.]
 for lab in ids:
  q=labs==lab; ts,b=m._fit(tr,y,ok & ~q); pred=tr.matrix(np.log(ts))@b; w=learn((lab,)); k=t>=np.median(t)
  print('LOLO',i,lab,'learned blend',w,'complete curve-drift/late residual MAE for archived, midpoint, nested',[(round(np.median(np.abs(np.median((y[q&ok,:-1]-u[q&ok,:-1])[:,k]/t[k],axis=1))),6), round(np.median(np.abs((y[q&ok,:-1]-u[q&ok,:-1])[:,k])),6)) for u in (update(pred,y[:,-1],ok,a) for a in (.25,.5,w))])
 ts,b=m._fit(tr,y,ok); pred=te.matrix(np.log(ts))@b; ptest=d['test']; complete=np.isfinite(ptest['coupon_features_raw']).all(axis=1); tl=np.asarray(ptest['lab_id']); w=learn(())
 for a in (.25,.5,w):
  u=update(pred,np.asarray(ptest['instant_rate_obs']),complete,a); e=u[:,:-1]+np.asarray(ptest['strain_0'])[:,None]-g['te_clean']; er=u[:,-1]-g['te_true_rate0']; slices=[tl==lab for lab in np.unique(tl)]; cs=[q&complete for q in slices if (q&complete).any()]
  print('public',i,'blend',a,'pooled strain/rate',np.round([rmse(e),rmse(er)],6),'worst lab strain/rate',np.round([max(rmse(e[q]) for q in slices),max(rmse(er[q]) for q in slices)],6),'worst complete lab late strain',round(max(rmse(e[q][:,t>=np.median(t)]) for q in cs),6))
