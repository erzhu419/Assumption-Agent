import importlib.util, json, pickle
import numpy as np
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; raw=c['bridge_raw']; j=np.arange(4); opp=m._OPPOSITE
peak=np.max(abs(raw),axis=1); clean=peak<4*np.median(peak); ids=np.flatnonzero(clean); folds=np.array_split(ids[np.argsort(np.linalg.norm(x[ids],axis=1),kind='stable')],6)
labels=['affine','trigonometric','trig + pairwise','trig + rational']; residuals={k:[] for k in labels}; inverse=[]; total_inv=[]
for idx in folds:
 fit=np.setdiff1d(ids,idx); b=m._bridge({'true_channel_loads':x[fit], 'bridge_raw':raw[fit]}); f,_=b
 def design(z,k):
  a=m._basis(z,f)
  if k==1: return a
  u=z; v=z[:,opp]; delta=u-v; pair=np.stack((u*v, delta, (u+v)**2, delta**3, u*u*v, u*v*v),-1)
  if k==2: return np.concatenate((a,pair),axis=-1)
  return np.concatenate((a,np.stack((u/(1+v*v), v/(1+u*u), delta/(1+delta*delta), (u+v)/(1+delta*delta)),axis=-1)),axis=-1)
 af=np.c_[np.ones(len(fit)),x[fit]]; aa=np.c_[np.ones(len(idx)),x[idx]]
 residuals['affine'].append(aa@np.linalg.lstsq(af,raw[fit],rcond=None)[0]-raw[idx]);
 for k,key in enumerate(labels[1:],1):
  af=design(x[fit],k); aa=design(x[idx],k)
  res=np.column_stack([aa[:,i]@np.linalg.lstsq(af[:,i],raw[fit,i],rcond=None)[0] for i in j])-raw[idx]
  residuals[key].append(res)
 err=m._inverse(raw[idx],b)-x[idx]; inverse.append(err); total_inv.append(err.sum(1));
for key,rr in residuals.items():
 print(key,'worst held-out fold/channel forward RMSE',max(np.sqrt(np.mean(r*r,axis=0)).max() for r in rr), 'pooled RMSE',np.sqrt(np.mean(np.concatenate(rr)**2)))
print('calibration out-of-fold inverse worst channel RMSE',max(np.sqrt(np.mean(e*e,axis=0)).max() for e in inverse),'worst total stress RMSE',max(np.sqrt(np.mean(e*e)) for e in total_inv),'max absolute total stress error',max(np.max(abs(e)) for e in total_inv))
assert max(np.sqrt(np.mean(e*e)) for e in total_inv)<0.01
tr,te=d['train'],d['test']; n=len(tr['strain_0']); X=m._features(tr,te); dt=d['past_dt']; t=d['eval_times']; b=m._bridge(c); past,hold=m._loads(tr,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs'];
for lab in np.unique(tr['lab_id']):
 sel=tr['lab_id']==lab; other=~sel; tau,w=m._fit(X[:n][other],past[other],hold[other],dt,t,y[other],rate[other]); a,r=m._design(tau,X[:n][sel],past[sel],hold[sel],dt,t); ec=a@w-y[sel]; er=r@w-rate[sel]; print('lab',lab,'OOF signed medians strain/rate',np.median(ec),np.median(er),'rate sign agreement',np.mean(np.sign(r@w)==np.sign(rate[sel])))
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); means=np.stack([np.mean(row['facility_telemetry_trace'],axis=0) for row in log]);
for i,row in enumerate(log): print('completed',row.get('completed_campaign_short_name'),'facility means',means[i].round(3),'distance to latest',np.linalg.norm(means[i]-means[-1]).round(3))
out=m.solve_batch(d); assert all(np.isfinite(v).all() for v in out.values()); print('isolated process outputs', {k:v.shape for k,v in out.items()})
