exec(open('/tmp/readout_search.py').read().split('c=s[')[0])
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular
rows=np.arange(n); rng=np.random.default_rng(42); radius=np.linalg.norm(x-x.mean(0),axis=1); edge=np.argsort(radius)[-12:]; splits=[(rows[:52],rows[52:])]+[(p[:52],p[52:]) for p in (rng.permutation(n) for _ in range(10))]+[(np.setdiff1d(rows,edge),edge)]
M=banks['base'][2,:n,:n]; R=banks['base'][1,:n,:n]; F=quad(x); y=Y[:,0]; C=(x@x.T)**3/6; C4=(x@x.T)**4/24

def universal(K,tr,te,F=F):
 a=np.linalg.cholesky(K[np.ix_(tr,tr)]); A=solve_triangular(a,F[tr],lower=True); b=solve_triangular(a,y[tr],lower=True); coef=np.linalg.lstsq(A,b,rcond=None)[0]; dual=solve_triangular(a.T,b-A@coef,lower=False); return F[te]@coef+K[np.ix_(te,tr)]@dual

def rational(tr,te):
 a=np.linalg.cholesky((.5*(M+R)+.002*np.eye(n))[np.ix_(tr,tr)]); b=solve_triangular(a,y[tr],lower=True)
 def basis(v): return F/(1+(x@v)**2)[:,None]
 def residual(v,output=False):
  f=basis(v); A=solve_triangular(a,f[tr],lower=True); w=np.linalg.lstsq(A,b,rcond=None)[0]; r=b-A@w
  if output: return f[te]@w
  return np.r_[r,.15*v]
 starts=np.vstack((.4*np.eye(4),[.2,.2,.2,.2],[-.2,.2,-.2,.2])); p=[least_squares(residual,v,max_nfev=50) for v in starts]; return residual(min(p,key=lambda z:z.cost).x,True)

def report(name,fn):
 es=np.array([np.mean(abs(fn(tr,te)-y[te])) for tr,te in splits]); print(name,np.round([es[0],es[1:-1].mean(),es[1:-1].max(),es[-1]],6))
report('baseline',lambda tr,te:universal(M+.001*np.eye(n),tr,te)); report('rational',rational)
for name,K in [('matern',M),('rbf',R),('mix',.5*(M+R))]:
 for w in (0,.03,.3,3):
  report((name,w),lambda tr,te:universal(K+w*(C+.25*C4)+.002*np.eye(n),tr,te))
