import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(suppress=True,precision=3,linewidth=145)
def fit(a,v):return np.linalg.lstsq(a,v,rcond=None)[0]
def choices(u):
 def poly(d):return np.column_stack([np.prod(u[:,i],axis=1) for i in itertools.combinations_with_replacement(range(4),d)])
 yield 'affine',u
 yield 'diag2',np.c_[u,u*u];yield 'diag3',np.c_[u,u*u,u**3];yield 'odd3',np.c_[u,u**3];yield 'cross2',np.c_[u,poly(2)];yield 'oddpair',np.c_[u,*[u[:,i]*u[:,j]**2 for i,j in itertools.permutations(range(4),2)],u**3]
 for w in [1.25,2.15,3,4,5,6.28,8,10,12]:
  for n,a in [('sin',np.sin(w*u)),('cos',np.cos(w*u)),('sc',np.c_[np.sin(w*u),np.cos(w*u)])]:
   yield f'{n}{w}',np.c_[u,a]
 for w in [.22,.4,.6,1,1.25,2.15,3,4]:
  for n,a in [('tanh',np.tanh(w*u)),('rational',u/(1+w*u*u)),('abs',abs(u))]:
   yield f'{n}{w}',np.c_[u,a]
for p in ['panel_a','panel_b','panel_c']:
 u=s[p]['u'];
 for k,v in s[p].items():
  if k=='u':continue
  idx=np.arange(len(u)); a=np.c_[np.ones(len(u)),u]; c=fit(a,v)
  if p=='panel_a':
   c=least_squares(lambda c:(a@c.reshape(-1,4)-v).ravel(),c.ravel(),loss='cauchy',f_scale=.05).x.reshape(-1,4); e=v-a@c; idx=np.flatnonzero(np.linalg.norm(e,axis=1)<1); print('A raw data clean',np.c_[u,v,e][idx][:18])
  scores=[]
  for n,a in choices(u[idx]):
   a=np.c_[np.ones(len(idx)),a]; c=fit(a,v[idx]);e=v[idx]-a@c;h=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,1);sc=np.sqrt(np.mean((e/np.maximum(1-h[:,None],1e-8))**2)); scores.append((sc,n,np.sqrt(np.mean(e**2))))
  print(p,k,sorted(scores)[:12]); print('linear residual',np.sqrt(np.mean((v[idx]-np.c_[np.ones(len(idx)),u[idx]]@fit(np.c_[np.ones(len(idx)),u[idx]],v[idx]))**2,0)))
