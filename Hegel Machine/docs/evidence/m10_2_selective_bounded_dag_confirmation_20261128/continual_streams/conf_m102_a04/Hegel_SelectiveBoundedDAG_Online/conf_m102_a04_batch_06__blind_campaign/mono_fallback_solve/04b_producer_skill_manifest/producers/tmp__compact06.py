import numpy as np
from scipy.optimize import least_squares
from itertools import product
G=np.load('/tmp/feat06.npz'); H=np.load('/tmp/d06.npz'); z,Y=G['z'],G['Y']; c,d=H['c'],H['d']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
_,sv,vt=np.linalg.svd(Y[:52,1:],full_matrices=False); T=np.c_[Y[:,0],Y[:,1:]@vt[:3].T]; S=T[:52].std(0)
base=np.c_[np.ones(len(z)),z*np.roll(c,1,axis=1),c*d]; p=c[:,0]+c[:,2]; q=c[:,1]+c[:,3]; r=d[:,0]-d[:,3]; s=d[:,1]-d[:,2]; v=z[:,2]+z[:,3];
def ff(a):
 k,l,m,n,o=a
 return np.c_[base,np.sin(k*p),np.sin(l*q),r/(1+m*r*r),s/(1+n*s*s),np.cos(o*v)]
def metrics(F,idx=np.arange(52),val=np.arange(52,72)):
 B=ls(F[idx],T[idx]); P=F@B; E=abs(np.c_[P[:,0],P[:,1:]@vt[:3]]-Y); return np.round(np.r_[E[idx,0].mean(),E[idx,1:].mean(),E[val,0].mean(),E[val,1:].mean()],6),B
for a in [[1.25,1.25,.22,.22,1.25],[1,1,.12,.12,1]]:
 print('initial',a,metrics(ff(a))[0]);
for weight in [np.ones(4),S,S**.5]:
 def res(a):
  F=ff(a)[:52]; return ((F@ls(F,T[:52])-T[:52])/weight).ravel()
 fit=least_squares(res,[1.25,1.25,.22,.22,1.25],bounds=([.05,.05,0,0,.05],[5,5,3,3,5]),gtol=1e-11); A=ff(fit.x); stats,B=metrics(A); print('fit',fit.x,stats,'coeff\n',np.round(B,4)); R=T-A@B
 for k,h in [('z',z),('c',c),('d',d)]:
  L=[]
  for i in range(4):
   for j in range(i,4):
    for a in [1,-1, .5,-.5, .8,-.8,1.25,-1.25]:
     x=h[:,i]+a*h[:,j]
     for tag,f in [('sin',np.sin(1.25*x)),('cos',np.cos(1.25*x)),('rat',x/(1+.22*x*x)),('tanh',np.tanh(1.25*x)),('p2',x*x)]:
      F=np.c_[A,f]; Q=F[:52]@ls(F[:52],T[:52])-T[:52]; L.append((np.mean((Q/S)**2),tag,i,j,a,metrics(F)[0].tolist()))
  print(k,sorted(L)[:4]);
