from graph import *
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from itertools import combinations
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=13).split(X)]
V=np.linalg.svd(T[:52],full_matrices=False)[2][:2]; targets=np.c_[Y,T@V.T]
F=PolynomialFeatures(2).fit_transform(X); coef=ls(F[:52],targets[:52]);print('raw quadratic coefficients',coef); print('scalar ranges and correlations with state modes',np.quantile(Y,[0,.25,.5,.75,1]),np.corrcoef(targets.T)[0]);
results=[]
for n,d in routes(X):
 z,w=C1(d),C2(d)
 for key,f in [('d',d),('C1',z),('C2',w),('C12',np.c_[z,w]),('dC12',np.c_[d,z,w]),('dC2',np.c_[d,w]),('C1dC2',np.c_[z,d+w]),('d+C1+C2',d+z+w)]:
  f=f/f.std(axis=0); D=cdist(f,f)**2/f.shape[1]
  P=PolynomialFeatures(2).fit_transform(d)
  for ell in [.4,.8,1.6,3.2]:
   r=np.sqrt(5*D)/ell; K=(1+r+r*r/3)*np.exp(-r)
   for lam in [.0003,.003,.03]:
    es=[]
    for i,j in splits:
     A=np.linalg.solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),np.c_[P[i],Y[i],T[i]]); p=A[:,:P.shape[1]]; b=A[:,P.shape[1]:]; c=ls(P[i].T@p,P[i].T@b); h=P[j]@c+K[np.ix_(j,i)]@(b-p@c); E=abs(h-np.c_[Y,T][j]);es.append([E[:,0].mean(),E[:,1:].mean()]);
    results.append((n,key,ell,lam,np.array(es)))
for col in range(2):
 print('kernel ranking for',col)
 for n,k,l,lam,e in sorted(results,key=lambda a:np.max(a[4][:,col]))[:10]:print(n,k,l,lam,np.round(e[:,col],5))
# Project linear/quadratic models through a single saturating scalar link.
for label,F in [('raw',PolynomialFeatures(2).fit_transform(X))]+[(n+'C',np.c_[PolynomialFeatures(2).fit_transform(d),C1(d),C2(d)]) for n,d in routes(X)]:
 scores=[]
 for link in ['identity','tanh','sin','rational','softsign','cubic']:
  ee=[]
  for i,j in splits:
   x=F[i]; y=Y[i]; b0=ls(x,y)
   def apply(z,c):
    a=np.exp(np.clip(c,-9,3))
    if link=='tanh':return np.tanh(a*z)/a
    if link=='sin':return np.sin(a*z)/a
    if link=='rational':return z/(1+a*z*z)
    if link=='softsign':return z/(1+a*abs(z))
    if link=='cubic':return z+c*z**3
    return z
   if link=='identity':pred=F[j]@b0
   else:
    def err(b):return apply(x@b[:-1],b[-1])-y
    r=least_squares(err,np.r_[b0,-1 if link!='cubic' else 0],max_nfev=100);pred=apply(F[j]@r.x[:-1],r.x[-1]);
   ee.append(abs(pred-Y[j]).mean());
  scores.append((max(ee),link,np.round(ee,4)))
 print('link',label,sorted(scores,key=lambda a:a[0])[:2])
