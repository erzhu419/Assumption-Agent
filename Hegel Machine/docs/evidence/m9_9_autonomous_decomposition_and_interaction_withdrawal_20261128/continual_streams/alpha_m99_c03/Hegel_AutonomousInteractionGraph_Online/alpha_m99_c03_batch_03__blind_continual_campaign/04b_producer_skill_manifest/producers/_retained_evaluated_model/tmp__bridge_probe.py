import json,pickle,itertools,numpy as np
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/archive_apparatus_log.json'
for r in json.load(open(p)):
 print('apparatus', {k:v for k,v in r.items() if 'trace' not in k and len(str(v))<400})
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; axis=(x!=0).sum(axis=1)<=1
print('axis',np.where(axis)[0]); print('raw range',[(k, v['past_channel_raw'].min(axis=(0,1)).round(2),v['hold_channel_raw'].max(axis=0).round(2)) for k,v in d.items() if k in ('train','test')]); print('axis sweeps',np.c_[x[axis],y[axis]].round(3))

def terms(kind):
 return [()]+[(i,) for i in range(4)]+[t for deg in range(2,kind+1) for t in itertools.combinations_with_replacement(range(4),deg)]
def feat(x,ts):
 return np.column_stack([np.prod(x[:,t],axis=1) for t in ts])
def fit(A,Y):
 # Remove readings far outside the bridge's observed operating range before fitting.
 ok=np.max(abs(Y),axis=1)<6.
 b=np.linalg.lstsq(A[ok],Y[ok],rcond=None)[0]
 for _ in range(20):
  e=np.linalg.norm(Y-A@b,axis=1); s=max(.005,np.median(e[ok])); w=(1+(e/(2*s))**2)**-1 * ok
  b=np.linalg.lstsq(A*np.sqrt(w[:,None]),Y*np.sqrt(w[:,None]),rcond=None)[0]
 return b
for deg in (1,2,3):
 ts=terms(deg); a=feat(x,ts); b=fit(a,y); e=np.linalg.norm(y-a@b,axis=1)
 rng=np.random.default_rng(17); groups=np.empty(len(x),int); groups[rng.permutation(len(x))]=np.arange(len(x))%5
 ep=np.zeros_like(y)
 for f in range(5):
  m=groups!=f; ep[~m]=y[~m]-a[~m]@fit(a[m],y[m])
 good=np.max(abs(y),axis=1)<6
 print('degree',deg,'resid quantiles',np.quantile(e,[0,.5,.9,.95,1]).round(4),'oof channel RMSE by slice',[(int(m.sum()),np.sqrt(np.mean(ep[m]**2,axis=0)).round(4)) for m in (good&axis,good&~axis)])
 for t,v in zip(ts,b):
  if max(abs(v))>.025: print(t,v.round(4))
 print('biggest errors',[(int(i),round(e[i],3)) for i in np.argsort(e)[-12:]])
