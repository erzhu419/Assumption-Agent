import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; N=52
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
af=lambda x: np.c_[np.ones(len(x)),x]
a=s['panel_a']; d=af(a['u']); h=fit(d,a['v']); k=np.argsort(np.linalg.norm(d@h-a['v'],axis=1))[:41]; h=fit(d[k],a['v'][k]); z=(x-h[0])/np.diag(h[1:]); b=s['panel_b']; H=fit(af(b['u']),b['v']); w=af(z)@H
c=s['panel_c']; u=c['u']; Cs={}
from itertools import combinations_with_replacement,product
from scipy.optimize import minimize_scalar
T=s['train']['traj']; I=np.arange(N); J=np.arange(N,len(y))
for name,q in [('cal',z),('B',w),('Blinear',w-H[0]),('Binv',(z-H[0])@np.linalg.inv(H[1:]))]:
 A=af(q); Q=np.linalg.qr(A[I])[0]; R=y[I]-Q@(Q.T@y[I]); X=q.T
 def score(cols):
  F=np.column_stack([A,*cols]); p=fit(F[I],y[I]); return np.std(F[I]@p-y[I]),np.mean(abs(F[J]@p-y[J])),p
 D={str(i):a for i,a in enumerate(X)}
 for i,j in combinations(range(4),2):
  for a in (-2,-1,-.5,.5,1,2): D[f'({i}{a:+g}*{j})']=X[i]+a*X[j]
 atoms={}; add=lambda k,v:atoms.setdefault(k,v)
 for i,j in combinations_with_replacement(range(4),2):
  p=X[i]*X[j]; add(f'{i}*{j}',p)
  for k,t in D.items():
   add(f'{i}*{j}*{k}',p*t)
   for a in (.1,.2,.5,1.):
    add(f'{i}*{j}/(1+{a}*{k}^2)',p/(1+a*t*t)); add(f'{i}*{j}*{k}/(1+{a}*{k}^2)',p*t/(1+a*t*t))
 for k,t in D.items():
  for a in (.2,.5,1.,1.7):
   for op,fun in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('sat',lambda x:x/(1+abs(x)))]: add(f'{op}({a}*{k})',fun(a*t))
  for i,j in combinations_with_replacement(range(4),2):
   for a in (.2,.5,1.):
    add(f'{i}*{j}*tanh({a}*{k})',X[i]*X[j]*np.tanh(a*t)); add(f'{i}*{j}*sin({a}*{k})',X[i]*X[j]*np.sin(a*t))
 names=list(atoms); V=np.stack(list(atoms.values()),axis=1); U=V[I]-Q@(Q.T@V[I]); U/=np.maximum(np.linalg.norm(U,axis=0),1e-11)
 ix=np.argsort(abs(U.T@R))[-35:][::-1]; best=[]
 for i in ix:
  u=U[:,i]; d=u@R; c=u@U; gain=np.abs(U.T@R-d*c)/np.sqrt(np.maximum(1-c*c,1e-10)); gain[abs(c)>.99999]=0
  j=np.argmax(gain); best.append((*score([V[:,i],V[:,j]])[:2],names[i],names[j]))
 print(name,'single',[(names[i],tuple(round(a,5) for a in score([V[:,i]])[:2])) for i in ix[:5]],'best pairs',sorted(best)[:8]);
 for base_name,B0 in [('aff',A),('quad',PolynomialFeatures(2).fit_transform(q))]:
  qb=np.linalg.qr(B0[I])[0]; r=y[I]-qb@(qb.T@y[I]); u=V[I]-qb@(qb.T@V[I]); u/=np.maximum(np.linalg.norm(u,axis=0),1e-11); used=[]
  for _ in range(6):
   i=np.argmax(abs(u.T@r)); used.append(i); f=np.c_[B0,V[:,used]]; p=fit(f[I],y[I]); r=y[I]-f[I]@p; print(base_name,names[i],round(np.std(r),5),round(np.mean(abs(y[J]-f[J]@p)),5))
