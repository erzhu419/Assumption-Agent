import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

def D(x):return np.c_[np.ones(len(x)),x]
def fit(F,y):return np.linalg.lstsq(F,y,rcond=1e-10)[0]
def poly(x):return D(np.c_[x,x*np.roll(x,-1,axis=1),x**3])
def get(s):
 tr=s['train']; x=np.r_[tr['x'],s['test']['x']]; u,v=(s['panel_a'][k] for k in ('u','v')); A=poly(u); keep=np.ones(len(u),bool)
 for _ in range(12):
  w=fit(A[keep],v[keep]); e=np.linalg.norm(A@w-v,axis=1); new=e<max(.001,3*np.median(e));
  if new.sum()<18 or np.array_equal(new,keep):break
  keep=new
 z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
 for i in range(20):
  K=w[5:9]; J=w[1:5]+np.roll(z,-1,axis=1)[:,:,None]*K+np.roll(z[:,:,None]*K,1,axis=1)+3*z[:,:,None]**2*w[9:13]; z-=np.clip(np.linalg.solve(J.transpose(0,2,1),(poly(z)@w-x)[...,None])[...,0],-.4,.4)
 u,v=(s['panel_b'][k] for k in ('u','v'))
 def H(x,q):return D(np.sign(x)*np.maximum(abs(x)-q[0],0)**q[1])
 def r(q):
  F=H(u,q);return (F@fit(F,v)-v).ravel()
 q=least_squares(r,[.35,1.5],bounds=([.1,.5],[.7,3])).x; b=H(z,q)@fit(H(u,q),v)
 p=s['panel_c']; u=p['u']; K=fit(poly(u),p['v1']); c=lambda x:poly(x)@K
 t=s['eval_times']; T=tr['traj'].T
 def F(q):return -np.expm1(-t[:,None]*q)
 def e(q):
  f=F(q);return (f@fit(f,T)-T).ravel()
 q=least_squares(e,[1.8,.42],bounds=([1.,.15],[3.,.85])).x; f=F(q); targets=np.c_[tr['y'],fit(f,T).T]
 return z,b,c,f,targets, (w,keep,x-poly(z)@w,u,p['v1'],p['v2'],q,e(q))

if __name__=='__main__':
 from probe23 import S
 z,b,c,f,Y,aux=get(S); w,keep,ie,u,v,h,q,te=aux
 print('A',keep.sum(),w.round(3),'inverse max',abs(ie).max(),'z range',z.min(),z.max(),'B',b.min(),b.max(),'C',fit(poly(u),v).round(3),'t',q,'temporal rmse',np.mean(te**2)**.5,'target std',Y.std(0));print('C ranges',u.min(),u.max(),v.min(),v.max(),h.min(),h.max()); n=52; N=len(Y)
 def forms(x):
  yield 'linear',x
  yield 'diag2',np.c_[x,x*x]
  yield 'cube',np.c_[x,x*x,x**3]
  for k in (.5,1,1.25,1.5,2.15,3):
   for nm,g in [('tanh',np.tanh),('erf',erf),('sin',np.sin),('soft',lambda v:v/np.sqrt(1+v*v)),('rational',lambda v:v/(1+v*v)),('asinh',np.arcsinh)]:
    a=g(k*x); yield f'{nm}{k}',np.c_[x,a,a*a]
 def pol2(x):return np.column_stack([x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])])
 def test(items,tar,n):
  rows=[]
  for nm,x in items:
   for nn,xx in forms(x):
    for pp,X in [('main',xx),('all2',np.c_[xx,pol2(x)])]:
     F=D(X); p=F[n:len(tar)]@fit(F[:n],tar[:n]); er=abs(p-tar[n:]); rows.append((np.mean(er[:,0]),np.mean(abs(er[:,1:]@f.T)) if tar.shape[1]==3 else np.mean(er),nm,nn,pp))
  for idx in [0,1]:print(sorted(rows,key=lambda a:a[idx])[:7]);
 print('C2 regression');test([('u',u),('v1',v),('u-v',u-v),('u+v',u+v),('u,v',np.c_[u,v])],h,19)
 print('Targets');test([('z',z),('b',b),('cz',c(z)),('cb',c(b)),('z,b',np.c_[z,b]),('z,cz',np.c_[z,c(z)]),('b,cz',np.c_[b,c(z)]),('b,cb',np.c_[b,c(b)]),('b+z',b+z),('b-z',b-z),('cz+b',c(z)+b),('cz-b',c(z)-b)],Y,n)
