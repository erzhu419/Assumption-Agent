import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
def affine(x):return np.c_[np.ones(len(x)),x]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[x[:,i].prod(1) for j in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),j)])
def cv(a,b,pri=True):
 p=a@fit(a[:52],b[:52]); err=np.abs(p-b); v=[err[:52].mean(),err[52:].mean()];
 if pri: print(np.round(v,5),end=' ')
 return v
print('times',t,'SV of T',np.linalg.svd(T,compute_uv=False),'x corr',np.corrcoef(x.T)); print('SV of residual after x');
for k in range(1,5):
 a=poly(x,k); cv(a,y);cv(a,T); print(k,np.linalg.svd(T-a@fit(a,T),compute_uv=False))
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; print(k,'u',u.min(0),u.max(0),'cov',np.linalg.eigvalsh(np.cov(u.T)))
 for key,v in p.items():
  if key=='u':continue
  for j in (1,2,3):
   a=poly(u,j); b=fit(a,v); e=np.linalg.norm(a@b-v,axis=1); print(key,j,'err',np.percentile(e,[0,25,50,75,90,100]),'loo med',np.median(e/np.maximum(1e-9,1-(np.linalg.svd(a,full_matrices=False)[0]**2).sum(1))))
   if j==1: print('coef',b,'outlier',np.round(e,3))
print('joint y/T correl',np.corrcoef(np.c_[y,T].T)[0]); print('y regress on x and traj sv'); u,_,v=np.linalg.svd(T[:52],full_matrices=False); w=T@v.T
for j in range(4):
 z=np.c_[x,w[:,:j]]; print(j,cv(affine(z),y,False),cv(poly(z,2),y,False))
print('train summary vs actual',tr['summary'],[y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std()])
