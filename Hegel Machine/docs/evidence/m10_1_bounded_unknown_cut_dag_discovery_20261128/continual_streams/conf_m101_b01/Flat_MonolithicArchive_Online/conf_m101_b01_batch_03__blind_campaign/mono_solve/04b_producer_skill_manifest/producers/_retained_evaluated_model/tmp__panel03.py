from discover03 import *
from scipy.special import erf
np.set_printoptions(precision=4,suppress=True,linewidth=120)
idx=np.flatnonzero(e<1); print('cal inliers',len(idx));
for name,uu,vv in [('A',u[idx],v[idx]),('C1',pc['u'],pc['v1']),('C2',pc['u'],pc['v2'])]:
 print(name)
 for name2,f in [('powers2',lambda x:np.c_[D(x),x*x]),('powers3',lambda x:np.c_[D(x),x*x,x**3]),('powers4',lambda x:np.c_[D(x),x*x,x**3,x**4]),('full2',lambda x:F(x,2)),('full2+diag3',lambda x:np.c_[F(x,2),x**3]),('full3',lambda x:F(x,3))]:
  X=f(uu); w=L(X,vv,rcond=None)[0]; err=X@w-vv; h=np.sum(X*np.linalg.pinv(X).T,1); print(name2, 'rms',np.sqrt(np.mean(err**2)),'loo',np.sqrt(np.mean((err/np.maximum(1-h[:,None],1e-8))**2)) if X.shape[1]<len(X) else None)
  if name2 in ['powers3','full2'] and name!='C2': print(w)
 for label,phi in [('tanh',np.tanh),('sin',np.sin),('erf',erf),('asinh',np.arcsinh)]:
  def mat(p,x): return D(phi(x*np.exp(p[0])+p[1:]))
  def fun(p):
   m=mat(p,uu); return (m@L(m,vv,rcond=None)[0]-vv).ravel()
  q=least_squares(fun,np.r_[-.4,np.zeros(4)],bounds=(np.r_[-4,np.full(4,-3)],np.r_[2,np.full(4,3)]),max_nfev=100); print(label,'in',np.sqrt(np.mean(q.fun**2)),q.x)
  def fout(p): return (phi(D(uu)@p.reshape(5,4))-vv).ravel()
  q=least_squares(fout,L(D(uu),vv,rcond=None)[0].ravel(),max_nfev=80);print(label,'out',np.sqrt(np.mean(q.fun**2)))
