import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; one=lambda x:np.c_[np.ones(len(x)),x]; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-9)[0]
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; U=p['u']; I=np.arange(4); P=np.stack([U[:,i]*U[:,j] for i in I for j in I if i<j],1)
 for name,V in p.items():
  if name=='u':continue
  keep=np.linalg.norm(V,axis=1)<4 if k=='panel_a' else np.ones(len(U),bool); u,v=U[keep],V[keep]; pp=P[keep]
  print(k,name,'n',sum(keep));
  opts={'linear':u,'poly2':np.c_[u,u*u,pp],'products':pp,'cyc':u*np.roll(u,-1,1),'tanh':np.tanh(u),'poly3':np.c_[u,u*u,u**3,pp]}
  for n,A in opts.items():
   A=one(A); print(n, np.std(A@ls(A,v)-v,axis=0))
  for tag,fn in [('sin',lambda u,q:np.sin(q*u)),('cos',lambda u,q:np.cos(q*u)),('tanh',lambda u,q:np.tanh(q*u)),('rat',lambda u,q:u/(1+(q*u)**2))]:
   for add in [False,True]:
    def res(q):
     A=one(np.c_[u,fn(u,q[0])] if add else fn(u,q[0]));return A@ls(A,v)-v
    q=least_squares(lambda q:res(q).ravel(),[2.15],bounds=(.1,6)).x; print(tag,add,q,np.std(res(q)))
t=s['eval_times'];T=s['train']['traj'][:52].T;print('svd',np.linalg.svd(T,compute_uv=False));
for rates in [[.42,1.8],[.42,1.8,3.5],[.22,2.15]]:
 def res(q):
  A=-np.expm1(-t[:,None]*q);return A@ls(A,T)-T
 q=least_squares(lambda q:res(q).ravel(),rates,bounds=(.01,10),gtol=1e-12).x;print('temporal',q,np.std(res(q)))
