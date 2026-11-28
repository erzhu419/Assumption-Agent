from modes import *
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.linear_model import Ridge
p=s['panel_c']; u=p['u']; V=np.c_[p['v1'],p['v2']]; P=aff(u); Bc=ls(P,V); R=V-P@Bc
print('panel affine',Bc,'resid SVD',np.linalg.svd(R,compute_uv=False));print('residual samples',np.c_[u,R][:12]);
def dictionary(v):
 a,b,c,d=v.T
 dirs=[*v.T,*[v[:,i]+sig*v[:,j] for i in range(4) for j in range(i+1,4) for sig in [-1,1]]]
 return np.c_[aff(v),*[f(d) for d in dirs for f in [np.abs,lambda x:x*x,lambda x:x**3,lambda x:np.sin(x),lambda x:np.cos(2*x),lambda x:np.tanh(2*x),lambda x:x/(1+x*x),lambda x:np.maximum(x-.35,0),lambda x:np.maximum(x+.35,0)]]]
def F(v,d):return PolynomialFeatures(d).fit_transform(v)
def calfit(kind,a):
 uu=p['u']; Q=np.r_[q,base['aib'],base['aibi'],x];
 if kind=='poly2': g=F(uu,2); gq=F(Q,2)
 if kind=='poly3': g=F(uu,3); gq=F(Q,3)
 if kind=='dict': g=dictionary(uu); gq=dictionary(Q)
 if kind=='rbf':
  K=np.exp(-a*cdist(uu,uu,'sqeuclidean')); M=np.exp(-a*cdist(Q,uu,'sqeuclidean')); return (aff(Q)@Bc+M@solve(K+1e-5*np.eye(len(uu)),R,assume_a='pos')).reshape(4,len(q),8)
 g-=0; sd=np.maximum(g.std(0),.1); model=Ridge(alpha=a).fit(g/sd,R); return (aff(Q)@Bc+model.predict(gq/sd)).reshape(4,len(q),8)
err=lambda P:np.r_[np.mean(abs(P[:,0]-y[n:])),np.mean(abs(P[:,1:]@H.T-T[n:]))]
results=defaultdict(list)
for d in [1,2,3,4]:
 for nm,v in base.items():
  G=F(v,d)
  for a in [1e-5,.001,.01,.1,1]:
   model=Ridge(alpha=a).fit(G[:n],Z[:n]); E=err(model.predict(G[n:]));results['base',nm].append((E,(d,a)))
for kind in ['poly2','poly3','dict','rbf']:
 for a in [1e-4,.003,.05,.3,1.]:
  for nm,v,G in zip(['ai','aib','aibi','x'],[q,base['aib'],base['aibi'],x],calfit(kind,a)):
   for mode,w in enumerate([G[:,:4],G[:,4:],G,np.c_[v,G[:,:4]],np.c_[v,G[:,4:]],np.c_[v,G]]):
    for d in [1,2,3]:
     X=F(w,d); X=X/np.maximum(X[:n].std(0),.1)
     for alpha in [.0001,.01,.1,1]:
      m=Ridge(alpha=alpha).fit(X[:n],Z[:n]);E=err(m.predict(X[n:]));results[kind,nm].append((E,(a,mode,d,alpha)))
for key,val in results.items():
 print(key,'joint',min(val,key=lambda r:max(r[0])),'scalar',min(val,key=lambda r:r[0][0]),'trajectory',min(val,key=lambda r:r[0][1]))
