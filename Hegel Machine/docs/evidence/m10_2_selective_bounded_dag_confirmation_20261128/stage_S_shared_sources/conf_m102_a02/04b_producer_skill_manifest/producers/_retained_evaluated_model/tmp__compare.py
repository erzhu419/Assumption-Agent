import numpy as np,pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; h=s['train']['traj']; t=s['eval_times']; z=s['test']['x']; np.set_printoptions(precision=6,suppress=True)
q=list(combinations_with_replacement(range(4),2)); A=np.column_stack([np.ones(len(x)),x]+[x[:,i]*x[:,j] for i,j in q]); print('times',t); print('quad order',q); print('y coeff',np.linalg.lstsq(A[:52],y[:52],rcond=None)[0]); W=np.linalg.lstsq(A[:,:5],h,rcond=None)[0]; print('linear traj coef',W); print('svd coef',np.linalg.svd(W,compute_uv=False)); print('svd traj',np.linalg.svd(h,compute_uv=False)); print('noise by time',np.std(h-A[:,:5]@W,axis=0));
def metrics(i,j):
  r=[]
  for k in [None,.002,.005,.01,.02]:
    c=np.linalg.lstsq(A[i],y[i],rcond=None)[0]; keep=np.ones(15,bool) if k is None else np.r_[np.ones(5,bool),abs(c[5:])>k]; c=np.linalg.lstsq(A[i][:,keep],y[i],rcond=None)[0]; r.append(np.mean(abs(A[j][:,keep]@c-y[j])))
  for d in [5,15]:
    g=A[i,:d]; w=np.linalg.lstsq(g,h[i],rcond=None)[0]; p=g@w; U,S,V=np.linalg.svd(p,full_matrices=False);
    for rank in [2,3,4,5,8]:
      v=V[:rank]; e=abs((A[j,:d]@w@v.T)@v-h[j]); r.extend([e.mean(),e.mean(axis=0).max()]);
  return r
ix=np.arange(len(x)); sets=[ix[52:]]; rng=np.random.default_rng(555); sets+=list(np.array_split(rng.permutation(ix),4)); sets += [np.argsort(abs(x[:,k]-x[:,k].mean()))[-15:] for k in range(4)]; sets += [np.argsort(np.linalg.norm(x,axis=1))[-15:]]
res=np.array([metrics(np.setdiff1d(ix,j),j) for j in sets]); print('ordered',res[0]); print('random worst',res[1:5].max(0)); print('boundary worst',res[5:].max(0)); print('validator tail:')
