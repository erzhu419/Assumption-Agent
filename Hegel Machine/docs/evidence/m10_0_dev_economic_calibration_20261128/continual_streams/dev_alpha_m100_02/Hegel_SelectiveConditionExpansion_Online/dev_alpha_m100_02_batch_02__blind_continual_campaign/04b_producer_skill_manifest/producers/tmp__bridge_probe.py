import pickle, numpy as np
from itertools import combinations_with_replacement
from pathlib import Path
D=pickle.loads(Path('/holdout_task/public_batch.pkl').read_bytes())[0]['dataset']; c=D['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; n=len(x)

def design(x,terms):
 return np.column_stack([np.prod(x[:,t],axis=1) for t in terms])

def fit(a,b):
 w=np.ones(len(a))
 for j in range(40):
  p=np.linalg.lstsq(a*w[:,None],b*w[:,None],rcond=None)[0]
  r=np.linalg.norm(a@p-b,axis=1)
  s=max(np.median(r)*1.5,1e-5)
  w=1/np.sqrt(1+(r/s)**2)
 return p,r

if __name__=='__main__':
 terms=[t for k in range(5) for t in combinations_with_replacement(range(4),k)]
 for label, t in [('linear',terms[:5]),('quadratic',terms[:15]),('quad+pure-cube',[j for j in terms if len(j)<=2 or (len(j)==3 and len(set(j))==1)]),('all-cube',terms[:35]),('cube+pure-quartic',[j for j in terms if len(j)<=3 or len(set(j))==1])]:
  a=design(x,t); p,r=fit(a,y); pred=y.copy()
  for h in range(n):
   k=np.arange(n)!=h; pred[h]=a[h]@fit(a[k],y[k])[0]
  cv=np.linalg.norm(pred-y,axis=1); good=r<1
  print(label,'gross',np.flatnonzero(~good),'fit',np.quantile(r[good],[.5,.9,1]),'LOO',np.quantile(cv[good],[.5,.9,1]),'axis/mixed max',cv[:16][good[:16]].max(),cv[16:][good[16:]].max())
  if label in ('quad+pure-cube','all-cube'):
   print([(j,np.round(v,5).tolist()) for j,v in zip(t,p) if np.max(np.abs(v))>.001])
 print('axis rows')
 print(np.round(np.c_[x[:16],y[:16]],5))
