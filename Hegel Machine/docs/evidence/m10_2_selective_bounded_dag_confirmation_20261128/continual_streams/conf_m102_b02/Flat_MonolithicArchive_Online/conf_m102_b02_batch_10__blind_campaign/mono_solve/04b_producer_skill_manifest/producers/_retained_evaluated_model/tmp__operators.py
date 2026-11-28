import numpy as np
from scipy.optimize import least_squares
from itertools import combinations
I=lambda u:np.column_stack((np.ones(len(u)),u))
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-11)[0]
i,j=np.triu_indices(4,1)
pair=lambda u:u[:,i]*u[:,j]
fun={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'rat':lambda z:z/(1+z*z),'cos':np.cos}
def profile(A,V,loo=False):
 c=ls(A,V); r=A@c-V
 if loo:
  d=1-np.sum(A*np.linalg.pinv(A).T,axis=1); r=r/np.maximum(d,.002).reshape((-1,)+(1,)*(V.ndim-1))
 return r.ravel()
def fit(A,V,seeds,loo=True):
 sol=min((least_squares(lambda q:profile(A(q),V,loo),p,bounds=(.08,8),max_nfev=70) for p in seeds),key=lambda r:r.cost)
 return np.sqrt(np.mean(sol.fun**2)),sol.x,ls(A(sol.x),V)
def calibrate(p,x,verbose=False):
 u,v=p['u'],p['v']; keep=np.ones(len(u),bool)
 for _ in range(6):
  a=I(u); r=np.linalg.norm(a@ls(a[keep],v[keep])-v,axis=1); keep=r<max(.002,3*np.median(r))
 u,v=u[keep],v[keep]; k=(np.arange(4)+2)%4
 def design(a,n,m,q):
  b=a[:,k]; F=[np.ones_like(a),a]; F+=[a**3] if n=='cube' else [fun[n](q[0]*a)]; F+=[b,b**3] if m=='cube' else [fun[m](q[1]*b)]; return np.stack(F,axis=-1)
 def fitcal(n,m):
  def co(d):return np.column_stack([ls(d[:,h],v[:,h]) for h in range(4)])
  def res(q):
   d=design(u,n,m,q); c=co(d); r=np.einsum('nij,ji->ni',d,c)-v
   for h in range(4):r[:,h]/=1-np.sum(d[:,h]*np.linalg.pinv(d[:,h]).T,axis=1)
   return r.ravel()
  sl=least_squares(res,[1.25,2.15],bounds=(.08,8)); return np.std(sl.fun),n,m,sl.x,co(design(u,n,m,sl.x))
 cand=sorted((fitcal(n,m) for n in ['cube','sin','tanh','rat','asinh'] for m in ['cube','sin','tanh','rat','asinh','atan']),key=lambda c:c[0]); e,n,m,q,c=cand[0]
 if verbose:
  for r in cand[:6]:print('A',r)
 def forward(z):return np.einsum('nij,ji->ni',design(z,n,m,q),c)
 init=(x-c[0])/c[1]; return np.array([least_squares(lambda z:forward(z[None])[0]-r,a,gtol=1e-12).x for a,r in zip(init,x)])
def operators(s,verbose=False):
 u,v=s['panel_b']['u'],s['panel_b']['v']; cand=[]
 for n in ['sin','tanh','atan','asinh','rat']:
  for m in ['','sin','tanh','atan','asinh','rat']:
   def b(a,q,n=n,m=m):return I(np.column_stack((a,fun[n](q[0]*a),fun[m](q[1]*a))) if m else np.column_stack((a,fun[n](q[0]*a))))
   e,q,c=fit(lambda q:b(u,q),v,[[1.25,2.15],[2.15,1.25],[3,2]] if m else [[1.25],[2.15],[3]])
   cand.append((e,n,m,q,c,b))
 cand.sort(key=lambda z:z[0]);
 if verbose:
  for e,n,m,q,c,_ in cand[:10]:print('B',e,n,m,q, np.linalg.norm(c,axis=1).round(4))
 u=s['panel_c']['u']; v1=s['panel_c']['v1']; v2=s['panel_c']['v2']; full=ls(I(pair(u)),v1); active=np.linalg.norm(full[1:],axis=1)>.002
 f1=lambda a:I(pair(a)[:,active]); c1=ls(f1(u),v1)
 e,q,c2=fit(lambda q:I(np.c_[u,np.cos(q[0]*u)]),v2,[[1.25],[2.15],[3]])
 if verbose: print('C2',e,q,c2)
 return [(n+'+'+m,lambda a,b=b,q=q,c=c:b(a,q)@c) for _,n,m,q,c,b in cand[:5]],lambda a:f1(a)@c1,lambda a:I(np.c_[a,np.cos(q[0]*a)])@c2
if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; z=calibrate(s['panel_a'],x,True); print('Z',z.min(),z.max()); operators(s,True)
