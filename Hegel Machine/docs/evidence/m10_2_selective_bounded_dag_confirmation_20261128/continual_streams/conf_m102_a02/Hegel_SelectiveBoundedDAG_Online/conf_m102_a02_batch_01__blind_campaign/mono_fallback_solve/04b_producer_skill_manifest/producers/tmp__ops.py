import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def design(x,d=1): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,idx],axis=1) for n in range(1,d+1) for idx in cr(range(x.shape[1]),n)])
def ls(f,v):return np.linalg.lstsq(f,v,rcond=None)[0]
def cv(f,v,ix):
 j=np.ones(len(f),bool);j[ix]=False;return np.mean(abs(v[ix]-f[ix]@ls(f[j],v[j])))
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[p]['u'];v=s[p][k]; print(p,k)
 for tag,f in [('poly2',design(u,2)),('powers3',np.c_[design(u,2),u**3]),('aff+pures23',np.c_[design(u),u**2,u**3]),('aff+pair cubes',np.c_[design(u),u**3,*[(u[:,i]+sg*u[:,j])**3 for i in range(4) for j in range(i+1,4) for sg in (-1,1)]]),('aff+triple',np.c_[design(u),*[np.prod(u[:,idx],axis=1) for idx in cr(range(4),3) if len(set(idx))==3]])]:
  print(tag,np.linalg.norm(v-f@ls(f,v)),cv(f,v,np.arange(18,24))); 
  if tag=='poly2':print(ls(f,v).T)
 for label,g in [('soft',lambda x:x/(1+abs(x))),('rat',lambda x:x/(1+x*x)),('tanh',np.tanh),('sin',np.sin),('asinh',np.arcsinh)]:
  def resid(a):
   f=design(g(u*a[:4]+a[4:]));return (f@ls(f,v)-v).ravel()
  o=least_squares(resid,np.r_[np.ones(4)*.7,np.zeros(4)],max_nfev=150); print('pre',label,np.linalg.norm(o.fun),o.x)
  X=design(u)
  def fit(j):
   b=ls(X,v[:,j]);
   def r(z):return z[0]*g(X@z[2:])+z[1]-v[:,j]
   o=least_squares(r,np.r_[1.5,0,b/1.5],max_nfev=300);return np.linalg.norm(o.fun),o.x
  out=[fit(j) for j in range(4)]; print('post',label,[q[0] for q in out]);
  if max(q[0] for q in out)<.02:print(np.array([q[1] for q in out]))
