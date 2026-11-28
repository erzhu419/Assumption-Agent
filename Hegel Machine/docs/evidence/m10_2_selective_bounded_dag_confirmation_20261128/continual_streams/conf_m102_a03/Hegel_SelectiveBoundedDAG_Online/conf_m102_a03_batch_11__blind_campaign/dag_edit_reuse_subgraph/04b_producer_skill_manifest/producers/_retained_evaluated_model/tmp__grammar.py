import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations,product
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(a,b):return np.linalg.lstsq(a,b,rcond=1e-11)[0]
def af(a):return np.c_[np.ones(len(a)),a]
def search(u,v):
 for j in range(4):
  a=u[:,j]; b=u[:,(j+1)%4]; c=u[:,(j+2)%4]; d=u[:,(j+3)%4]; A=af(u); V=v[:,j]
  F={}; tele=(.22,1.25,2.15)
  for k,x in enumerate((a,b,c,d)):
   F.update({f'sq{k}':x*x,f'cu{k}':x**3})
   for w in tele:
    F.update({f'atan{k}:{w}':np.arctan(w*x),f'tanh{k}:{w}':np.tanh(w*x),f'sin{k}:{w}':np.sin(w*x),f'rat{k}:{w}':x/(1+w*x*x),f'ratc{k}:{w}':x**3/(1+w*x*x)})
  for k,l in combinations(range(4),2): F[f'p{k}{l}']=u[:,k]*u[:,l]
  q=np.linalg.qr(A)[0]; M=np.column_stack(list(F.values())); M-=q@(q.T@M); M/=np.maximum(1e-13,np.linalg.norm(M,axis=0)); y=V-q@(q.T@V); ss=M.T@y; g=M.T@M; err=(y@y)-(ss[:,None]**2+ss[None,:]**2-2*g*ss[:,None]*ss[None,:])/np.maximum(1-g*g,1e-14); err[np.tril_indices(len(F))]=np.inf
  print(j,[(list(F)[k],list(F)[l],round(np.sqrt(max(err[k,l],0)/len(u)),8)) for k,l in map(lambda o:np.unravel_index(o,err.shape),np.argsort(err.ravel())[:8])]);print('best single',[(list(F)[k],round(np.sqrt(max(y@y-ss[k]**2,0)/len(u)),8)) for k in np.argsort(-abs(ss))[:5]])
for p,ks in [('panel_a',['v']),('panel_b',['v']),('panel_c',['v1','v2'])]:
 for key in ks:
  u=S[p]['u'];v=S[p][key]; ok=np.linalg.norm(v,axis=1)<5;print(p,key,u.shape); search(u[ok],v[ok]);
  A=af(u[ok]); print('affine coef',np.round(fit(A,v[ok]),5));print('res affine',np.linalg.norm(A@fit(A,v[ok])-v[ok],axis=0)/np.sqrt(len(A)))
