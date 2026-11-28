import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement as C
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def af(x):return np.c_[np.ones(len(x)),x]
def ls(f,y):return np.linalg.lstsq(f,y,rcond=None)[0]
def loo(f,y):
 q=np.linalg.svd(f,full_matrices=False)[0][:,:np.linalg.matrix_rank(f)]; h=np.sum(q*q,axis=1);return np.sqrt(np.mean(((y-f@ls(f,y))/np.maximum(1-h,1e-8)[:,None])**2))
def pol(x,d):return np.column_stack([x[:,j].prod(1) for k in range(d+1) for j in C(range(x.shape[1]),k)])
a=S['panel_a']; A=ls(af(a['u']),a['v']);
for _ in range(10):
 ix=np.argsort(np.linalg.norm(af(a['u'])@A-a['v'],axis=1))[:45]; A=ls(af(a['u'][ix]),a['v'][ix])
b=S['panel_b']; B=ls(af(b['u']),b['v']); c=S['panel_c'];u=c['u'];
def extra(x,k,g):
 if k=='sine':return np.sin(g*x)
 if k=='cos':return np.cos(g*x)
 if k=='tanh':return np.tanh(g*x)
 if k=='soft':return x/np.sqrt(1+g*x*x)
 if k=='sq':return x*x
 if k=='rat':return x*x/(1+g*x*x)
 if k=='exp':return np.exp(-g*x*x)
 if k=='cube':return x**3
 if k=='quart':return x**4
 if k=='abs':return abs(x)
for kk in ['v1','v2']:
 y=c[kk];print('panel',kk); res=[]
 for base in [u,af(u)@B,np.roll(u,1,axis=1)*u,np.roll(u,2,axis=1)*u,np.roll(u,1,axis=1)+u,np.roll(u,1,axis=1)-u]:
  for k in ['sine','cos','tanh','soft','sq','rat','exp','cube','quart','abs']:
   for g in [.22,.5,1,1.25,1.5,2,2.5,3,4,5]:
    f=np.c_[af(u),extra(base,k,g)]; res.append((loo(f,y),k,g, np.sqrt(np.mean((f@ls(f,y)-y)**2)), np.array_equal(base,u),np.array_equal(base,af(u)@B)))
 print(sorted(set(res))[:16]);print('resid svd',np.linalg.svd(y-af(u)@ls(af(u),y),compute_uv=False));
 for p in [af(u),np.c_[af(u),u*u],np.c_[af(u),u*u,u**3],np.c_[af(u),u*u,u**3,u**4]]: print('additive LOO',loo(p,y))
x=S['train']['x']; y=S['train']['y']; T=S['train']['traj']; q=T@np.linalg.svd(T[:52],full_matrices=False)[2][0]; print('y quantile',np.quantile(y,[0,.05,.5,.95,1]));print('cor y,q',np.corrcoef(y,q)[0,1]);
xs={'raw':x,'A':(x-A[0])/A[1:].diagonal()};Z={}
for n,w in xs.items():
 Z[n]=w; Z[n+'B']=af(w)@B
 for k in ('v1','v2'):
  for d in (1,2):Z[n+k+str(d)]=pol(w,d)@ls(pol(u,d),c[k])
for j,h in enumerate((y,q)):
 res=[]
 for n,w in Z.items():
  for k,g in [('raw',0)]+[(k,g) for k in ['sine','tanh','soft'] for g in [.22,.5,1,1.5,2]]:
   w2=w if k=='raw' else extra(w,k,g)
   for d in [1,2,3,4]:
    f=pol(w2,d); f=f/np.maximum(f[:52].std(0),.15)
    for inv in ([0,1.25,1.5,2] if j==0 else [0]):
     hh=np.arctanh(np.clip(h/inv,-.999,.999)) if inv else h
     for pen in [.003,.04,.25]:
      h2=f@np.linalg.solve(f[:52].T@f[:52]+pen*np.diag(np.r_[0,np.ones(f.shape[1]-1)]),f[:52].T@hh[:52]); h2=inv*np.tanh(h2) if inv else h2
      res.append((np.abs(h2[52:]-h[52:]).mean(),n,k,g,d,inv,pen,np.std(h2[:52]-h[:52])))
 print('best',j,*sorted(res)[:12],sep='\n')
