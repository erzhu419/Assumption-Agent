import pickle,itertools
import numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; n=len(x)
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(u,v):return np.linalg.lstsq(aff(u),v,rcond=None)[0]
p=s['panel_a']; g=np.linalg.norm(p['u']-p['v'],axis=1)<1; A=fit(p['u'][g],p['v'][g]); p=s['panel_b']; B=fit(p['u'],p['v']);
def a(z):return aff(z)@A
def ai(z):return (z-A[0])@np.linalg.inv(A[1:])
def b(z):return aff(z)@B
def bi(z):return (z-B[0])@np.linalg.inv(B[1:])
def amp(r):
 f=-np.expm1(-r*t); z=tr@f/(f@f); return z,np.mean((tr-z[:,None]*f)**2)
r=minimize_scalar(lambda r:amp(r)[1],bounds=(.3,1.4),method='bounded').x; z=amp(r)[0]; T=np.c_[y,z];
def quad(x):return np.c_[aff(x),*[x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)]]
def evaluate(D,Y=T):
 c=np.linalg.lstsq(D[:52],Y[:52],rcond=None)[0]; return np.mean(abs(D[52:]@c-Y[52:]),axis=0)
from scipy.optimize import least_squares
print('ranges',T.min(0),T.max(0),'times',t,'telemetry',s['telemetry_vec']); print('train summary',s['train']['summary'], 'actual',T.mean(0),T.std(0))
D=aff(x); P=quad(x); it=np.triu_indices(4); rng=np.random.default_rng(14)
splits=[(np.arange(k),np.arange(k,n)) for k in (52,56)]+[(p[:52],p[52:]) for p in [rng.permutation(n) for _ in range(3)]]

def trial(i,j,kind,pen):
 U=np.linalg.qr(D[i])[0]; R=T[i]-U@(U.T@T[i]); scale=R.std(0); R/=scale
 def get(h):
  g=D@h.reshape(-1,5).T
  if kind in ('prod','sprod','aprod','sqp'):
   fun={'prod':np.tanh,'sprod':np.sin,'aprod':np.arctan,'sqp':lambda v:v/(1+v*v)**.5}[kind]; g=fun(g); N=g[:,::2]*g[:,1::2]; return np.c_[N,g]
  if kind=='satur':return np.c_[np.tanh(g),np.tanh(g)**2]
  return np.c_[np.tanh(g),g*np.tanh(g)]
 def solve(h):
  N=get(h)[i]; N-=U@(U.T@N); b=np.linalg.solve(N.T@N+pen*np.eye(N.shape[1]),N.T@R); return N,b
 def err(h):
  N,b=solve(h); return np.r_[(N@b-R).ravel(),np.sqrt(pen)*b.ravel(),.018*h]
 best=None
 co=np.linalg.lstsq(P[i],T[i],rcond=None)[0][5:]
 eig=[]
 for c in [co[:,0],co[:,1],co.mean(1)]:
  m=np.zeros((4,4)); m[it]=c; m=(m+m.T)/2; l,V=np.linalg.eigh(m); eig.append((V[:,0]+V[:,-1],V[:,0]-V[:,-1]))
 for k in range(12):
  H=rng.normal(0,.45,(4,5)); H[:2,1:]+=np.array(eig[k%3])*.7
  res=least_squares(err,H.ravel(),bounds=(-4,4),max_nfev=115,ftol=5e-5)
  if best is None or res.cost<best.cost:best=res
 N=get(best.x); b=solve(best.x)[1]*scale; a=np.linalg.lstsq(D[i],T[i]-N[i]@b,rcond=None)[0]; Q=D@a+N@b
 return np.mean(abs(Q[j]-T[j]),axis=0),np.mean(abs(Q[i]-T[i]),axis=0),best.x.reshape(-1,5)
for mode in ('prod','sprod','aprod','sqp','satur','mult'):
 for pen in (.0003,.008):
  o=[trial(i,j,mode,pen) for i,j in splits[:2]]; print(mode,pen,np.round([p[0] for p in o],5),o[0][1].round(5),o[0][2].round(2),flush=True)
