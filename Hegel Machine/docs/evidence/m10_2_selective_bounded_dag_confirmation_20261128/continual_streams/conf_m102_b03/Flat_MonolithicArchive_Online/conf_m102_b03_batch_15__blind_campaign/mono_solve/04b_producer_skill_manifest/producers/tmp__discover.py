import itertools as it,pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u=s['panel_a']['u'];v=s['panel_a']['v'];n=len(u);D=lambda z:np.c_[np.ones(len(z)),z];ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0];rng=np.random.default_rng(2)
for kind in ('affine','local','smooth','cubic','quadratic','expanded'):
 out=[]
 for j in range(4):
  a=u[:,j];opts=[]
  for r in (1,2,3):
   b=u[:,(j+r)%4];f={'affine':u,'local':np.c_[a,b,a*b,a**3],'smooth':np.c_[a,np.tanh(1.8*a),np.sin(2.4*b)],'cubic':np.c_[u,u**3],'quadratic':np.c_[u,*[u[:,i]*u[:,k] for i,k in it.combinations_with_replacement(range(4),2)]],'expanded':np.c_[u,u**3,*[u[:,i]*u[:,k] for i,k in it.combinations(range(4),2)]]}[kind];f=D(f);y=v[:,j]
   def score(c):return np.sort(abs(f@c-y))[int(n*.72)]
   c=min((ls(f[g],y[g]) for g in (rng.choice(n,f.shape[1]+2,False) for _ in range(350))),key=score)
   for _ in range(3):
    e=abs(f@c-y);g=e<max(.001,4*np.median(e));c=ls(f[g],y[g]);
   opts.append((score(c),r,np.round(c,4),np.quantile(abs(f@c-y),[.5,.8,.9])))
  out.append(min(opts,key=lambda a:a[0]))
 print(kind,out)
t=s['eval_times'];T=s['train']['traj'][:52]
for baseline in [False,True]:
 def basis(k):
  H=-np.expm1(-t[:,None]*k);return D(H) if baseline else H
 def err(k):
  H=basis(k);return (H@ls(H,T.T)-T.T).ravel()
 p=least_squares(err,[2,.4],bounds=([.6,.03],[5,1.5])); print('time',baseline,p.x,np.sqrt(np.mean(p.fun**2)))
