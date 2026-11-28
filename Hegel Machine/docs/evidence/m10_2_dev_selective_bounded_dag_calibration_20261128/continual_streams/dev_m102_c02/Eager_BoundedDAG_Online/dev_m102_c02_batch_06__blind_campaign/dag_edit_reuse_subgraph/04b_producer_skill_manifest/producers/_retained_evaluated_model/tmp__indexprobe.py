exec(open('/tmp/probe.py').read().split("print('A row error'")[0])
from scipy.optimize import least_squares
from scipy.special import expit
rng=np.random.default_rng(7); A=np.c_[y,t[:,-1]]
def fit(z,kind):
 L=np.c_[np.ones(len(z)),z]; Q=np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)]); X=np.c_[L,Q] if kind.endswith('q') else L
 def feat(w):
  h=L@w[:5]
  if kind.startswith('sin'): k=np.sin(h)
  elif kind.startswith('hinge'): k=np.maximum(h,0)
  elif kind.startswith('abs'): k=np.sqrt(h*h+.02)
  elif kind.startswith('soft'): k=np.logaddexp(0,h)
  elif kind.startswith('sig2'): return np.c_[X,expit(h),expit(L@w[5:])] if len(w)==5 else np.c_[X,expit(L@w[:5]),expit(L@w[5:])]
  elif kind.startswith('tanh'): k=np.tanh(h)
  elif kind.startswith('rat'): k=1/(1+h*h)
  else: k=h**3
  return np.c_[X,k]
 def fun(w):
  F=feat(w)[:52]; R=F@np.linalg.lstsq(F,A[:52],rcond=1e-9)[0]-A[:52]; return np.r_[R.ravel(),0.001*w]
 m=10 if kind.startswith('sig2') else 5
 opts=[least_squares(fun,w,max_nfev=90) for w in rng.normal(0,1.7,(14,m))]; best=min(opts,key=lambda o:o.cost); F=feat(best.x); B=np.linalg.lstsq(F[:52],Y[:52],rcond=1e-9)[0]; E=abs(F[52:]@B-Y[52:]); return best.cost,E[:,0].mean(),E[:,1:].mean(),best.x
print('target SVD',np.linalg.svd(Y,compute_uv=False),flush=True)
for name,z in [('c',c),('u',u),('B1.25',n._b_design(u,1.25)@np.linalg.lstsq(n._b_design(b['u'],1.25),b['v'],rcond=None)[0])]:
 print('COORD',name,flush=True)
 for k in ['sina','sinq','hingea','hingeq','absa','softa','tanha','rata','ratq','cubea','sig2a','sig2q']:
  e=fit(z,k); print(k,np.round(e[:3],5), np.round(e[3],3),flush=True)
