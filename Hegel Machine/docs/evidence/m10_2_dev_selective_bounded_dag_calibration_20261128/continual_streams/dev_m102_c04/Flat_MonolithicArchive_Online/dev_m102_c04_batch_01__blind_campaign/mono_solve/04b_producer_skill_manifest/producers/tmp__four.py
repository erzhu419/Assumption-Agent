from dag import *
from scipy.optimize import minimize_scalar
vv=np.c_[y,t]; print('y from trajectory',np.mean(abs(poly(t,1)@ls(poly(t[:52],1),y[:52])-y)[52:]),'y mean std',y.mean(),y.std()); print('temporal sv',np.linalg.svd(t,compute_uv=False));
def cv(f,b):
 w=ls(f[:52],b[:52]); e=abs(f@w-b);return e[:52].mean(0),e[52:].mean(0)
r=[]
for n,z in paths(x,2).items():
 for base,f0 in [('N',np.ones((72,1))),('L',poly(x,1)),('P',poly(x,2)),('C',np.c_[poly(x,1),Q(z),H(z)])]:
  for m in ['s','c','both','tanh','phase']:
   def f(k):
    zz=k*z
    w={'s':np.sin(zz),'c':np.cos(zz),'both':np.c_[np.sin(zz),np.cos(zz)],'tanh':np.tanh(zz),'phase':np.sin(zz+.22*z*z)}[m];return np.c_[f0,w]
   for i,j in [(0,1),(1,9)]:
    b=vv[:,i:j]; grid=np.arange(.6,9,.3); errs=[cv(f(k),b)[0].mean() for k in grid]; o=np.argmin(errs); k=minimize_scalar(lambda k:cv(f(k),b)[0].mean(),bounds=(max(.2,grid[o]-.35),grid[o]+.35),method='bounded').x; a,c=cv(f(k),b);r.append((i,a.mean(),c.mean(),n,base,m,k))
for i in [0,1]:
 l=[a for a in r if a[0]==i]; print('train-best',*sorted(l,key=lambda x:x[1])[:15],sep='\n');print('test-best',*sorted(l,key=lambda x:x[2])[:15],sep='\n')
