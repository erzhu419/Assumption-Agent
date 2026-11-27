import pickle,itertools,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; n=len(x)
def design(x,powers):
 return np.column_stack([np.prod(x[:,p],axis=1) for p in powers])
def robust(a,b):
 w=np.ones(len(a))
 for _ in range(30):
  fit=np.linalg.lstsq(a*w[:,None], b*w[:,None],rcond=None)[0]
  e=np.linalg.norm(a@fit-b,axis=1)
  w=1/np.sqrt(1+(e/max(2.5*np.median(e),1e-9))**2)
 return fit
rng=np.random.default_rng(34)
folds=np.array_split(rng.permutation(n),5)
for degree in range(1,5):
 for sep in ([False,True] if degree>1 else [False]):
  powers=[p for d in range(degree+1) for p in itertools.combinations_with_replacement(range(4),d) if not sep or len(set(p))<=1]
  a=design(x,powers)
  if len(powers)>50:continue
  pred=np.empty_like(y)
  for val in folds:
   train=np.setdiff1d(np.arange(n),val)
   pred[val]=a[val]@robust(a[train],y[train])
  e=np.linalg.norm(pred-y,axis=1)
  f=robust(a,y); ef=np.linalg.norm(a@f-y,axis=1)
  print('degree, separable',degree,sep,'oof percentiles',np.round(np.percentile(e,[25,50,75,85,90,95,100]),7),'fit percentiles',np.round(np.percentile(ef,[50,85,90,95]),7),'spike rows',np.where(ef>2)[0])
  if degree==3 and not sep:
   print('cubic coefficients',[(p,np.round(v,5).tolist()) for p,v in zip(powers,f)])
