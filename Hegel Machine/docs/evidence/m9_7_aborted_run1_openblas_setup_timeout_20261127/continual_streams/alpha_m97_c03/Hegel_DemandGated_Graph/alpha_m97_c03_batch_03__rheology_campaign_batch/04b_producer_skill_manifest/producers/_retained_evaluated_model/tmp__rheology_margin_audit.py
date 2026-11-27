import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c03_batch_03_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_03.pkl','rb'))[0]['dataset']
a,b=d['train'],d['test']; lab=a['lab_id']; miss=np.isnan(a['coupon_features_raw']).any(axis=1)
g=m._calibrate(d['calibration_load_cells']); x=m._align(a,b)[:len(lab)]; past,h=m._physical(a,g); dt,t=d['past_dt'],d['eval_times']; y=np.column_stack((a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']))
for mode in ('all','complete'):
 e=np.empty_like(y)
 for c in range(4):
  use=(lab!=c)&((~miss) if mode=='complete' else True); z=lab==c
  tau,w=m._fit(past[use],h[use],dt,t,x[use],y[use]); e[z]=m._design(tau,past[z],h[z],dt,t,x[z])@w-y[z]
  print(mode,c,'tau',np.round(tau,5),'p95 strain, rate rmse',np.quantile(abs(e[z,:-1]),.95),np.sqrt(np.mean(e[z,-1]**2)),'complete median signed by horizon',np.round(np.median(e[z&~miss],axis=0),4))
  # Compare the entire design, including rate, against a direct recurrence and scalar constitutive terms.
  mem=np.zeros((z.sum(),2))
  for step,stress in zip(dt,past[z].T):
   q=np.exp(-step/tau); mem=q*mem+(1-q)*stress[:,None]
  q0,q1=(h[z,None]-mem).T
  v=w[2]+x[z]@w[3:7]+w[7]*np.sign(h[z])*np.sqrt(abs(h[z]))
  f0=w[0]*q0+w[8]*(q0-q1); f1=w[1]*q1
  strain=f0[:,None]*(1-np.exp(-t/tau[0]))+f1[:,None]*(1-np.exp(-t/tau[1]))+v[:,None]*t/(1+.85*t)
  rate=f0/tau[0]+f1/tau[1]+v
  np.testing.assert_allclose(np.column_stack((strain,rate)),e[z]+y[z],atol=1e-12)
 for name,z in [('complete',~miss),('missing',miss),('nonnegative stress',h>=0),('negative stress',h<0)]:
  if z.any(): print(mode,name,z.sum(),'med abs',np.median(abs(e[z,:-1])),np.median(abs(e[z,-1])))
cal=d['calibration_load_cells']; errors=[]
for c in range(4):
 rows=np.flatnonzero(cal['lab_id']==c)
 from itertools import combinations
 for group in combinations(rows,3):
  selected=np.flatnonzero((cal['lab_id']!=c)|np.isin(np.arange(len(lab))[:len(cal['lab_id'])],group))
  base={k:v[selected] for k,v in cal.items()}
  for idx in np.flatnonzero(base['lab_id']==c):
   for noise in (-6.,6.):
    alt=dict(base,true_stress=base['true_stress'].copy()); alt['true_stress'][idx]+=noise
    errors.append(np.max(abs(m._calibrate(alt)[c]-g[c])))
print('Three-point sweeps with one EMI spike: max coefficient error and failures above 0.001:',max(errors),sum(v>.001 for v in errors),'of',len(errors))
