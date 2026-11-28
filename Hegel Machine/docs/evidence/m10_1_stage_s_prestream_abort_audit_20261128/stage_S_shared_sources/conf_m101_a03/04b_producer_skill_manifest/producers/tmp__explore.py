import numpy as np,pickle,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']
np.set_printoptions(precision=5,suppress=True,linewidth=160)
def poly(x,d):
 return np.column_stack([np.prod(x[:,ix],axis=1) for k in range(d+1) for ix in it.combinations_with_replacement(range(x.shape[1]),k)])
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
print('times',t,'summary discrepancy',s['train']['summary']-[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]);print('traj singular values',np.linalg.svd(z,compute_uv=False));print('x covariance',np.cov(x.T))
for p in ('panel_a','panel_b','panel_c'):
 u=s[p]['u']; A=poly(u,1)
 for k,v in s[p].items():
  if k=='u':continue
  m=ls(A,v)
  for _ in range(8):
   i=np.argsort(np.linalg.norm(v-A@m,axis=1))[:int(.85*len(u))];m=ls(A[i],v[i]);
  print(p,k,'coef',m,'residuals',np.sort(np.linalg.norm(v-A@m,axis=1)))
  for d in (1,2):
   a=poly(u,d); r=np.array([v[i]-a[i]@ls(np.delete(a,i,0),np.delete(v,i,0)) for i in range(len(u))]);print('LOO degree',d,np.quantile(np.linalg.norm(r,axis=1),[.25,.5,.8,.95,1]))
for d in (1,2,3):
 a=poly(x,d); B=np.c_[y,z]; errs=[]
 for test in np.array_split(np.random.default_rng(3).permutation(len(x)),4):
  train=np.setdiff1d(np.arange(len(x)),test);errs.append(np.mean(abs(a[test]@ls(a[train],B[train])-B[test]),axis=0))
 print('train poly d',d,'mae mean/max',np.mean(errs,0),np.max(errs,0))
print('delta sample',np.diff(np.c_[np.zeros(len(x)),z],axis=1)[:10]);print('corr',np.corrcoef(np.c_[x,y,z].T)[:5,4:])
