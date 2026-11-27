import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
d=item['dataset']; tr,te=d['train'],d['test']; q=item['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; b=m._bridge(d['calibration_load_cells']); X=m._features(tr,te)[:len(tr['lab_id'])]; p,h=m._loads(tr,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; labs=tr['lab_id']; out=m.solve_batch(d); tau=out['estimated_taus']; th=out['hereditary_weights']; rms=lambda x: np.sqrt(np.mean(x*x))
for g in np.unique(labs):
 k=labs==g; a,w=m._fit(X[~k],p[~k],h[~k],dt,t,y[~k],r[~k]); F,G=m._design(X[k],p[k],h[k],dt,t,a); ec=(F@w).reshape(y[k].shape)-y[k]; er=G@w-r[k]; print('benchtop holdout lab',g,'worst horizon MAE, rate MAE, max relative spectrum shift',max(np.mean(abs(ec),axis=0)),np.mean(abs(er)),max(abs(a/tau-1)))
if q is not None:
 x=m._features(q,q)[:len(q['lab_id'])]; qp,qh=m._loads(q,b); lab=q['lab_id']; c,rc=m._chamber_prediction(q,x,b,t,tau,th); E=np.c_[q['strain_curves_obs']-c,q['instant_rate_obs']-rc]; T=np.r_[np.log1p(t),1.]; drift=th[2]+x@th[3:]; _,rate_design=m._design(x,qp,qh,dt,t,tau); memory=qh[:,None]-rate_design[:,:2]*tau; B=np.r_[-np.expm1(-t[:,None]/tau),(1/tau)[None,:]].T; contrast=qh-qp[:,-1]
 def score(F=None):
  o=E.copy()
  for g in np.unique(lab):
   k=lab==g
   if F is not None:
    coef=np.linalg.lstsq(F[~k].reshape(-1,F.shape[-1]),E[~k].ravel(),rcond=None)[0]; o[k]-=F[k]@coef
  return np.round([max(rms(o[lab==g,:-1]) for g in np.unique(lab)),max(rms(o[lab==g,-1]) for g in np.unique(lab)),max(rms(o[:,j]) for j in range(len(t)))],6)
 print('diagnostics: worst lab strain RMSE, worst lab rate RMSE, worst horizon RMSE')
 print('exact',score()); print('residual hereditary memory',score(memory[:,None,:]*B.T[None,:,:])); print('hold-last pairwise contrast',score((contrast[:,None]*T)[...,None]));
 for a in (.25,1.,4.):
  Q=drift/(1+a*contrast**2)-drift
  print('rational drift attenuation',a,score((Q[:,None]*T)[...,None]))
 for g in np.unique(lab):
  k=lab==g; print('diagnostic signed lab bias',g,E[k,:-1].mean(),E[k,-1].mean())
assert all(np.isfinite(v).all() for v in out.values()) and .05 < tau[0] < tau[1] < 25
