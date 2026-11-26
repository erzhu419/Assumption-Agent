import pickle, runpy
import numpy as np
m=runpy.run_path('/holdout_task/alpha_w02_batch_08_solver.py')
for e in pickle.load(open('/holdout_task/public_alpha_w02_batch_08.pkl','rb')):
 d=e['dataset']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; sy=d['yield_stress']; lab=tr['lab_id']; raw=tr['coupon_features_raw']; has=~np.isfinite(raw).all(1); n=len(lab)
 c=m['_calibrate'](d['calibration_load_cells']); f=m['_features'](tr,te)[:n]; a=m['_inputs'](tr,c,dt,sy,f)
 z=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; fold=np.empty(n,int); rng=np.random.default_rng(738)
 for l,s,h in np.unique(np.c_[lab,tr['stratum'],has],axis=0):
  q=np.flatnonzero((lab==l)&(tr['stratum']==s)&(has==h)); rng.shuffle(q); fold[q]=(np.arange(len(q))+rng.integers(4))%4
 def subset(q):return {k:v[q] for k,v in tr.items()}
 pred0=np.empty_like(z); pred1=np.empty_like(z)
 for k in range(4):
  q=fold!=k; train=subset(q); test=subset(~q); aa=tuple(x[q] for x in a); ab=tuple(x[~q] for x in a)
  taus,w=m['_parameters'](aa,raw[q],dt,t,z[q]); logs=np.log(taus)
  p=m['_design'](ab,dt,t,logs)@w; pred0[~q]=p
  fit_rate=m['_design'](aa,dt,t[:0],logs)[:,0]@w
  m['_condition_missing'](train,test,aa,p,(fit_rate,w[3:7]),t); pred1[~q]=p
 hot=(a[3]!=0)|(a[4]!=0)
 print('calibration',c.round(6).tolist()); print('yield/expected test state count',sy,np.count_nonzero(te['residual_plastic_state']))
 for name,q in [(f'lab{l}',lab==l) for l in range(4)]+ [('below_yield',~hot),('activated',hot),('missing',has),('complete',~has)]+[(f'lab{l}_missing',(lab==l)&has) for l in range(4)]:
  if not np.any(q):continue
  vals=[]
  for pred in (pred0,pred1):
   r=(z-pred)[q]; vals.append(np.round([np.nanmedian(abs(r[:,:-1])),np.sqrt(np.nanmean(np.minimum(abs(r[:,:-1]),.12)**2)),np.sqrt(np.nanmean(r[:,-1]**2))],6).tolist())
  print(name,int(q.sum()),'strain median / capped strain RMSE / rate RMSE',vals)
