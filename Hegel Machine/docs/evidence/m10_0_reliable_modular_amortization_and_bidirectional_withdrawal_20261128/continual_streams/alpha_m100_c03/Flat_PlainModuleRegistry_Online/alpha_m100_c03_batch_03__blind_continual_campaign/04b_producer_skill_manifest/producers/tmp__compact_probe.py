import numpy as np,pickle
from itertools import combinations_with_replacement
C=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x=C['true_channel_loads']; y=C['bridge_raw']
terms=[()] + [t for d in (1,2,3) for t in combinations_with_replacement(range(4),d)]
A=np.column_stack([np.prod(x[:,t],axis=1) for t in terms]); m=np.count_nonzero(x,axis=1)
def fit(a,v):
 w=np.ones(len(a))
 for _ in range(40):
  b=np.linalg.lstsq(a*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(a@b-v,axis=1); w=1/np.sqrt(1+(r/max(2*np.median(r),1e-12))**2)
 return b

def sparse(a,v,method):
 if method=='full': return fit(a,v)
 if method=='adaptive':
  b=fit(a,v); r=np.linalg.norm(a@b-v,axis=1); good=r<8*np.median(r); b0=b.copy(); b[:]=0
  for j in range(4):
   mask=np.abs(b0[:,j])>max(.001,12*np.median(np.abs((a@b0-v)[good,j]))); b[mask,j]=np.linalg.lstsq(a[good][:,mask],v[good,j],rcond=None)[0]
  return b
 masks=[]
 for j in range(4):
  n=(j+1)%4;p=(j-1)%4
  masks.append([terms.index(t) for t in [(),(j,),(n,),(p,),tuple(sorted((j,n))),(j,j,j)]])
 b=np.zeros((len(terms),4)); w=np.ones(len(a))
 for _ in range(25):
  for j,idx in enumerate(masks): b[idx,j]=np.linalg.lstsq(a[:,idx]*w[:,None],v[:,j]*w,rcond=None)[0]
  r=np.linalg.norm(a@b-v,axis=1); w=1/np.sqrt(1+(r/max(2*np.median(r),1e-12))**2)
 return b
b=sparse(A,y,'full'); good=np.linalg.norm(A@b-y,axis=1)<.01
print('gross spike rows:',np.flatnonzero(~good)); print('design counts:',np.unique(m,return_counts=True))
for method in ['full','adaptive','cyclic']:
 errs=[]
 for i in np.flatnonzero(good):
  idx=np.arange(len(x))!=i; pred=A[i]@sparse(A[idx],y[idx],method); errs.append((i,np.linalg.norm(pred-y[i])))
 i,e=np.array(errs).T; i=i.astype(int)
 print(method,'median/p95/max',np.quantile(e,[.5,.95,1]),'axes max', e[m[i]<=1].max(),'mixed max',e[m[i]>1].max())
print('paired sweep range',[(k,np.shape(v)) for k,v in C['single_channel_sweeps'].items()])
