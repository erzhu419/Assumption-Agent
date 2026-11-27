import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c02_batch_01_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]['dataset']
tr=d['train']; n=len(tr['lab_id']); cal=m._calibration(d); x=m._align(d); a=m._inputs(d,tr,cal,x[:n]); raw=tr['coupon_features_raw']; missing=np.isnan(raw).any(1); labs=tr['lab_id']; strata=tr['stratum']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']
def sub(mask): return tuple(v if j in (2,3) else v[mask] for j,v in enumerate(a))
def ae(v): return np.round([np.median(np.abs(v)),np.quantile(np.abs(v),.9)],5)
for lab in (0,1,2,3,-1):
 fit=labs!=lab; val=(~fit if lab>=0 else fit); aa=sub(val)
 for mode in ('all','complete_fit'):
  use=fit if mode=='all' else fit&~missing
  p=m._fit(sub(use),y[use],rate[use]); c,r,*_=m._design(p[:2],*aa)
  ec=c@p[2:]-y[val]; er=r@p[2:]-rate[val]; b=missing[val]
  print(lab,mode,'tau',np.round(np.exp(p[:2]),5),'complete',ae(ec[~b]),ae(er[~b]),'missing',ae(ec[b]),ae(er[b]))
  if mode=='all' and lab==-1:
   print('largest missing-rate residuals',np.sort(np.abs(er[b]))[-8:])
# Compare balanced mean and per-cell mean using actually observed feature values as masking targets.
rng=np.random.default_rng(17)
for rep in range(2):
 ix=np.flatnonzero(~missing); cols=rng.integers(4,size=len(ix)); targets=x[ix,cols]; vals=[]
 for group in ('balanced','cell'):
  pred=[]
  for i,j in zip(ix,cols):
   keep=(np.arange(n)!=i)&np.isfinite(raw[:,j]); means=np.array([[np.mean(x[:n][keep&(labs==l)&(strata==s),j]) for s in np.unique(strata)] for l in range(4)]); pred.append(means.mean() if group=='balanced' else means[labs[i],strata[i]])
  vals.append((group,ae(np.asarray(pred)-targets)))
 print('masked measured features',vals)
