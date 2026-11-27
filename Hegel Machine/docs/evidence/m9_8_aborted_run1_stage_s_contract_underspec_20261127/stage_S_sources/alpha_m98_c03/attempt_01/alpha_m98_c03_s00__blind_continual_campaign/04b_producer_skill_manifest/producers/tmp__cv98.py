import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; t=d['eval_times']; dt=d['past_dt']; cal=m._calibrate(d['calibration_load_cells']); bench=d['train']; test=d['test']; pilot=d['chamber_verification_trials']
def subset(s,mask): return {k:v[mask] for k,v in s.items()}
base={}; scores={}
for label,s,noise in [('bench',bench,.012),('pilot',pilot,.006)]:
 panel=m._panels(bench,test) if label=='bench' else m._panels(pilot)
 x=m._standardize(s,panel)
 for lab in np.unique(s['lab_id']):
  tr=s['lab_id']!=lab; va=~tr; ss=subset(s,tr); check=subset(s,va)
  for mode in ('local','train'):
   if mode=='train' and label=='bench':continue
   if mode=='local':
    tau,w,dr=m._fit_relaxation(ss,m._stress(ss,cal),dt,t,noise)
    fill=m._impute(x,s) if label=='pilot' else m._impute(np.r_[x,m._standardize(test,panel)],s,test)
    coef=m._fit_drift(m._design(x[tr],fill),dr,noise*.5)
    if label=='bench':base[lab]=tau,w,coef,fill
   else:tau,w,coef,fill=base[lab]
   z=m._kernel(tau,dt,t,*m._stress(check,cal))@w; drift=m._design(x[va],fill)@coef
   e=check['strain_0'][:,None]+z[:,:-1]+drift[:,None]*t-check['strain_curves_obs']; r=z[:,-1]+drift-check['instant_rate_obs']; miss=~np.isfinite(x[va]).all(1)
   a=tuple(round(v,5) for v in [np.median(abs(e)),np.quantile(abs(e),.9),np.mean(abs(r)),np.mean(abs(e[:,-1])),np.mean(abs(e[miss])) if miss.any() else 0]); print(label,int(lab),mode,'median strain / q90 strain / rate MAE / final MAE / missing MAE',a,'params',np.r_[tau,w,coef].round(4)); scores.setdefault((label,mode),[]).append(a)
print('worst by lab', {k:np.max(v,axis=0) for k,v in scores.items()})
rng=np.random.default_rng(98); change=np.eye(4)+rng.normal(0,.12,(4,4)); bias=rng.normal(0,.3,4); raw=lambda x:x@change+bias
for p in [pilot,None]:
 case=dict(d,chamber_verification_trials=p); r=m.solve_batch(case); assert all(np.all(np.isfinite(v)) for v in r.values()); assert .05<r['estimated_taus'][0]<r['estimated_taus'][1]<25
 transformed=dict(case,calibration_load_cells=dict(case['calibration_load_cells'],bridge_raw=raw(d['calibration_load_cells']['bridge_raw'])))
 for k in ('test','train','chamber_verification_trials'):
  if transformed[k] is not None: transformed[k]=dict(transformed[k],**{a:raw(transformed[k][a]) for a in ('past_channel_raw','hold_channel_raw')})
 q=m.solve_batch(transformed); print('bridge invariance',p is not None,max(np.max(abs(q[k]-r[k])) for k in r)); assert all(np.allclose(q[k],v,atol=2e-6,rtol=1e-6) for k,v in r.items());
 # Independent recurrence and t=0 derivative identity.
 tau=r['estimated_taus']; u,h=m._stress(test,cal); mem=np.zeros((len(h),2))
 for j,length in enumerate(dt):
  decay=np.exp(-length/tau); mem=mem*decay+u[:,j,None]*(1-decay)
 z=m._kernel(tau,dt,t,u,h); assert np.allclose(z[:,-1],(h[:,None]-mem)/tau); assert np.allclose(z[:,:-1],(h[:,None]-mem)[:,None,:]*(1-np.exp(-t[:,None]/tau)))
 at0=m.solve_batch(dict(case,eval_times=t))['initial_rate_pred']; print('valid mode',p is not None, np.r_[tau,r['hereditary_weights']].round(5),[v.shape for v in r.values()]); assert np.allclose(at0,r['initial_rate_pred'])
