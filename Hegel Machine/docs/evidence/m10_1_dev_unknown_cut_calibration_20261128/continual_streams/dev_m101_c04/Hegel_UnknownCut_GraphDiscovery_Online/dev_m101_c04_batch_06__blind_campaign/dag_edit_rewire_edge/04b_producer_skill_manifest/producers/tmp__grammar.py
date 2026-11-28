from dag_probe import *
from scipy.optimize import least_squares
c=C['u']; q=basis(c,1)
def vp(f):
 return lambda k: (f(c,k)@np.linalg.lstsq(f(c,k),v,rcond=None)[0]-v).ravel()
def tan(x,k): return np.tanh(x*k)
def rat(x,k): return x*x/(1+k*x*x)
for v,g in [(C['v1'],tan),(C['v2'],rat)]:
 for scalar in [True,False]:
  ff=lambda x,k:np.c_[basis(x,1),g(x,k)]
  k=least_squares(vp(ff),np.full(1 if scalar else 4,2. if g==tan else .2),bounds=(.001,10)).x
  w=np.linalg.lstsq(ff(c,k),v,rcond=None)[0]; print(g.__name__,np.round(k,4),np.mean(abs(ff(c,k)@w-v))); print(np.round(w,3))

def run(k=2.2,j=.2):
 q=basis(u,2); target=Y
 def check(F):
  e=[]
  for ix in [np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,60)]:
   it=np.setdiff1d(np.arange(72),ix); e.append(np.mean(abs(F[ix]@np.linalg.lstsq(F[it],target[it],rcond=1e-9)[0]-target[ix])))
  return tuple(np.round(e,5))
 D={'tu':tan(u,k),'tz':tan(z,k),'ru':rat(u,j),'rz':rat(z,j)}
 S={}
 for n,x in D.items():
  for m,a in [('u',u),('z',z)]:
   S[m+n]=a*x
   S[m+n+'all']=np.einsum('ni,nj->nij',a,x).reshape(len(a),-1)
 for a in D:
  for h in D:
   if a<=h: S[a+h]=D[a]*D[h]
 D.update(S)
 tests=[]
 def add(ks):
  F=np.c_[q,*[D[n] for n in ks]]; err=check(F); tests.append((max(err),err,ks))
 add([])
 for a in D: add([a])
 from itertools import combinations
 for ks in combinations(D,2):
  if sum(D[n].shape[1] for n in ks)<=24: add(ks)
 for e in sorted(tests,key=lambda row:row[0])[:18]:print(e)
 for e in sorted(tests,key=lambda a:a[1][0])[:5]:print('official',e)
run()
