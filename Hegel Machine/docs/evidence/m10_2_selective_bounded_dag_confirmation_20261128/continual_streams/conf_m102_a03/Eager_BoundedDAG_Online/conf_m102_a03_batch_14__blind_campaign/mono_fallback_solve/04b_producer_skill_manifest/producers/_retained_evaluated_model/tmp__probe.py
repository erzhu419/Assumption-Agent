import pickle, numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t=s['train']; x=t['x']; Y=np.c_[t['y'],t['traj']]; q=list(combinations_with_replacement(range(4),2))
def design(u,d=2):
 return np.c_[np.ones(len(u)),u,*[np.prod(u[:,i],axis=1) for k in range(2,d+1) for i in combinations_with_replacement(range(4),k)]]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
models={}
np.set_printoptions(precision=4,suppress=True,linewidth=145)
for p in ('panel_a','panel_b','panel_c'):
 u=s[p]['u']
 for k,v in s[p].items():
  if k=='u':continue
  for d in (1,2):
   a=design(u,d);idx=np.arange(len(u));c=fit(a,v)
   if p=='panel_a':
    from scipy.optimize import least_squares
    c=least_squares(lambda c:(a@c.reshape(-1,4)-v).ravel(),c.ravel(),loss='cauchy',f_scale=.02).x.reshape(-1,4)
    idx=np.where(np.linalg.norm(a@c-v,axis=1)<max(.08,3*np.median(np.linalg.norm(a@c-v,axis=1))))[0];c=fit(a[idx],v[idx])
   e=a[idx]@c-v[idx];h=np.sum(np.linalg.qr(a[idx],mode='reduced')[0]**2,axis=1);print(p,k,d,len(idx),'RMSE',np.sqrt(np.mean(e*e,axis=0)),'LOO',np.sqrt(np.mean((e/(1-h[:,None]))**2,axis=0)))
   if d==2:print(c);models[p+k]=c
A=models['panel_av']; z=np.array([least_squares(lambda z: (design(z[None])@A-v).ravel(),np.linalg.solve(A[1:5].T,v-A[0])).x for v in x]);print('inverse range',z.min(0),z.max(0))
ops={k:(lambda a,c=c:design(a)@c) for k,c in models.items() if k!='panel_av'}
F={'x':x,'z':z};
for n in ('x','z'):
 for k,op in ops.items():F[n+k]=op(F[n]);
for n in list(F)[2:]:
 for k,op in ops.items():F[n+k]=op(F[n]);
for d in (1,2):
 results=[]
 for n,f in F.items():
  a=design(f,d); yh=a@fit(a[:52],Y[:52]);e=np.mean(np.abs(yh[52:]-Y[52:]),0);results.append((e[0],e[1:].mean(),n))
 print('degree',d,'y',sorted(results)[:8]);print('traj',sorted(results,key=lambda r:r[1])[:8])
np.savez('/tmp/features.npz',**F)
