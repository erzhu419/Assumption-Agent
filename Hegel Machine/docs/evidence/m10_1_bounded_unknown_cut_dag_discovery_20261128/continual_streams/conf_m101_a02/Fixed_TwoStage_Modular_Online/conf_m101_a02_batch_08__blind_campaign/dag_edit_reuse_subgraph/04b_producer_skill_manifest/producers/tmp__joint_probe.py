exec(open('/tmp/readout_probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
n=len(u); B=s['panel_b']; k0,g=m._fit_b(B['u'],B['v']); Az=poly(z,1); Dz=poly(z,2); Du=poly(u,2); X=np.c_[u,z]; C=s['panel_c']; w=C['u']; f=lambda q:np.c_[poly(q,1),np.tanh(2.1*q)]; V1=f(u)@np.linalg.lstsq(f(w),C['v1'],rcond=None)[0]; V2=Du@np.linalg.lstsq(poly(w,2),C['v2'],rcond=None)[0]
T=np.c_[Y[:,0],Y[:,1:]/np.sqrt(8)]
def err(pred,b):
 e=np.mean(abs(pred-T[b]),0);return np.array([e[0],e[1:].mean()*np.sqrt(8)])
def ridge(D,a,b,lam):
 scale=np.maximum(np.linalg.norm(D[a],axis=0),1e-8); D=D/scale; return D[b]@np.linalg.solve(D[a].T@D[a]+lam*np.eye(D.shape[1]),D[a].T@T[a])
for name,d in [('z/u quadratic',np.c_[Dz,Du[:,1:]]),('z quadratic/u linear',np.c_[Dz,u]),('z/u mixed',poly(X,2)),('z quadratic plus z*u',np.c_[Dz,u,(z[:,:,None]*u[:,None,:]).reshape(n,16)]),('z quadratic + panel C',np.c_[Dz,V1,V2]),('z quadratic + panel C quad',np.c_[Dz,poly(V1,2)[:,1:],poly(V2,2)[:,1:]]),('z quadratic + measured',np.c_[Dz,x]),('z quadratic + u*z contrast',np.c_[Dz,u,u*z]),('z quadratic plus separate u cubic',np.c_[Dz,u,u*u,u**3]),('z quadratic + u higher even',np.c_[Dz,u,u**2,u**4,u**6])]:
 for lam in [1e-7,.001,.01]:
  e=np.array([err(ridge(d,a,b,lam),b) for a,b in fs]); print(name,lam,e[0].round(4),e.max(0).round(4))
for fn in [lambda q,k:q*(-np.expm1(-k*q*q)),lambda q,k:q**3/(1+k*q*q),lambda q,k:np.tanh(k*q),lambda q,k:np.sin(k*q),lambda q,k:np.sign(q)*abs(q)**k,lambda q,k:q*np.exp(k*q*q)]:
 for typ in range(4):
  def d(k):
   h=fn(u,k); return [np.c_[Dz,h],np.c_[Az,u,poly(h,2)[:,1:]],np.c_[Az,u,poly(h,2)[:,5:]],np.c_[Dz,u,poly(h,2)[:,5:]]][typ]
  ee=[]; kk=[]
  for a,b in fs:
   # Search on leave-one-out residuals strictly inside train fold.
   def loss(k):
    q=np.linalg.svd(d(k)[a],full_matrices=False)[0];r=(T[a]-q@(q.T@T[a]))/(1-(q*q).sum(1)[:,None]); return np.mean(r*r)
   grid=np.linspace(.08,3.5,20); j=np.argmin([loss(i) for i in grid]); opt=minimize_scalar(loss,bounds=(grid[max(0,j-1)],grid[min(j+1,len(grid)-1)]),method='bounded'); D=d(opt.x); kk.append(opt.x); ee.append(err(D[b]@np.linalg.lstsq(D[a],T[a],rcond=None)[0],b))
  print('nonlinear',fn.__code__.co_firstlineno,typ,np.round(kk,2),np.round(ee[0],4),np.max(ee,0).round(4))
# Residual pattern for dependence on u with products constructed in calibrated gain coordinates.
H=(z-g[0])@np.linalg.inv(g[5:]);E=T-Dz@np.linalg.lstsq(Dz,T,rcond=None)[0]; print('B coords',H[:5].round(3));print('largest feature correlations',[(j,float(np.corrcoef(E[:,0],v)[0,1])) for j,v in enumerate(np.c_[u,u*u,u**3,H,H**2,H**3].T)])
# Evaluate squared separations and norm-dependent saturation kernels as a check on unmodeled interactions.
for q in [u,z,np.c_[u,z],np.c_[u,V1,V2,z]]:
 R=cdist(q,q,'sqeuclidean'); best=[]
 for ell in [.25,.5,.8,1.2,2,4]:
  K=np.exp(-R/(2*ell**2));
  for lam in [1e-4,.002,.02,.2]:
   es=[]
   for a,b in fs:
    d=Dz[a]; solver=np.linalg.solve(K[np.ix_(a,a)]+lam*np.eye(len(a)),np.c_[d,T[a]]); c=np.linalg.lstsq(d.T@solver[:,:d.shape[1]],d.T@solver[:,d.shape[1]:],rcond=None)[0];es.append(err(Dz[b]@c+K[np.ix_(b,a)]@(solver[:,d.shape[1]:]-solver[:,:d.shape[1]]@c),b))
   best.append((max(np.max(es,0)),ell,lam,np.round(es[0],4),np.max(es,0).round(4)))
 print('kernel',q.shape[1],sorted(best,key=lambda t:t[0])[:2])
