import numpy as np,pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; b=sw['single_bank_sweeps']; bx=np.eye(4)[b['bank_id']]*b['ref_signal'][:,None]

def fit(a,y):
 c=np.linalg.lstsq(a,y,rcond=None)[0]
 for t in [1,.25,.06,.015,.004,.001,.00025,.00006]:
  for _ in range(10):
   w=1/(1+(np.linalg.norm(y-a@c,axis=1)/t)**2); u=np.sqrt(w[:,None]); c=np.linalg.lstsq(a*u,y*u,rcond=None)[0]
 return c

terms=[z for d in [2,3] for z in combinations_with_replacement(range(4),d)]
features={'aff':lambda q:np.c_[np.ones(len(q)),q], 'odd':lambda q:np.c_[np.ones(len(q)),q,q**3], 'diag':lambda q:np.c_[np.ones(len(q)),q,q**2,q**3], 'full':lambda q:np.c_[np.ones(len(q)),q,*[q[:,z].prod(1) for z in terms]]}
for key,f in features.items():
 a=f(x); e=np.zeros_like(y)
 for te in np.array_split(np.random.default_rng(13).permutation(len(x)),10):
  tr=np.setdiff1d(np.arange(len(x)),te); e[te]=a[te]@fit(a[tr],y[tr])-y[te]
 c=fit(a,y); ck=(abs(y).max(1)<8); pb=(f(bx)@c)[np.arange(len(bx)),b['bank_id']]; eb=pb-b['raw_readout']; print(key,'held out quartiles',np.quantile(abs(e[ck]),[.5,.9,.99,1]),'bank errors clean',np.round(eb,7));
 if key in ['odd','diag']: print('coefficients',np.round(c,7)); print('largest clean errors',np.round(np.sort(np.linalg.norm(e[ck],axis=1))[-12:],7))
print('study keys',list(s)); print('pulse diag?',pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0].get('campaign_diagnostics') is not None)
