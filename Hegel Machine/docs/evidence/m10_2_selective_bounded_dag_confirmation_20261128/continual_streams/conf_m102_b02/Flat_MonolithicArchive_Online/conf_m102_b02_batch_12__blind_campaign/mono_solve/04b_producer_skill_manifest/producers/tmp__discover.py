import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def I(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def rm(r):return np.mean(r*r)**.5
def fp(f,u,v):
 a=f(u);r=a@ls(a,v)-v; h=(np.linalg.svd(a,full_matrices=False)[0][:,:np.linalg.matrix_rank(a)]**2).sum(1)
 return rm(r),rm(r/(1-h[:,None]))
def calibrate(x):
 u,v=(s['panel_a'][k] for k in ('u','v'));k=np.linalg.norm(v,axis=1)<5
 def D(u):
  p=np.roll(u,-1,axis=1);return np.stack([np.ones_like(u),u,p,u*p,u**3],-1)
 a=D(u);c=np.stack([ls(a[k,j],v[k,j]) for j in range(4)]); print('A',rm(np.einsum('nij,ij->ni',a[k],c)-v[k]),np.round(c,4))
 return np.array([least_squares(lambda q:(D(q[None])[0]*c).sum(1)-r,r,gtol=1e-11).x for r in x])
F={'sin':lambda x,k:np.sin(k*x),'tanh':lambda x,k:np.tanh(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'rat':lambda x,k:x/(1+k*x*x),'expe':lambda x,k:1-np.exp(-k*x*x),'cos':lambda x,k:1-np.cos(k*x)}
def op(p,o,verbose=True):
 u,v=s[p]['u'],s[p][o]; best=[]
 def add(fn,starts,lo,hi,tag):
  def r(q):
   a=fn(u,q);return (a@ls(a,v)-v).ravel()
  fits=[least_squares(r,x,bounds=(lo,hi),max_nfev=100) for x in starts]; q=min(fits,key=lambda f:f.cost).x; f=lambda x:fn(x,q); score=fp(f,u,v); c=ls(f(u),v);best.append((score,tag,q,lambda x,f=f,c=c:f(x)@c))
 for name,g in F.items():
  add(lambda x,q,g=g:I(np.c_[x,g(x,q[0])]),[[i] for i in (.3,1.25,2.15,3.6)],[.05],[7],name+' free')
  add(lambda x,q,g=g:I(x+q[1]*g(x,q[0])),[[i,j] for i in (.3,1.25,2.15,3.6) for j in (-.4,.4)],[.05,-10],[7,10],name+' tied')
 if o!='v2':
  for k,l in combinations(list(F)[:4],2):
   g,h=F[k],F[l]
   for mode in ('free','tied','nonlinear'):
    def fn(x,q,g=g,h=h,mode=mode):
     a,b=g(x,q[0]),h(x,q[1]);return I(np.c_[x,a,b] if mode=='free' else (x+q[2]*a+q[3]*b if mode=='tied' else np.c_[a,b]))
    starts=[[i,j]+([.2,.4] if mode=='tied' else []) for i,j in [(2.15,1.25),(3.2,2.),(1.25,2.15)]]; add(fn,starts,[.15,.15]+([-6,-6] if mode=='tied' else []),[6,6]+([6,6] if mode=='tied' else []),(k,l,mode))
 best.sort(key=lambda q:q[0][1]);
 if verbose:print(p,o,[(np.round(b[0],6),b[1],np.round(b[2],4)) for b in best[:10]])
 return best[0][-1]
if __name__=='__main__':
 t=s['train'];z=calibrate(np.r_[t['x'],s['test']['x']]); B=op('panel_b','v'); C=op('panel_c','v1'); D=op('panel_c','v2'); w=B(z); np.savez('/tmp/coords.npz',z=z,w=w,c=C(z),d=D(z),h=C(w),j=D(w),bC=B(C(z)),bD=B(D(z)),bCD=B(C(z)+D(z)),CD=C(D(z)),DC=D(C(z)),Ch=C(D(w)),Dh=D(C(w)))
