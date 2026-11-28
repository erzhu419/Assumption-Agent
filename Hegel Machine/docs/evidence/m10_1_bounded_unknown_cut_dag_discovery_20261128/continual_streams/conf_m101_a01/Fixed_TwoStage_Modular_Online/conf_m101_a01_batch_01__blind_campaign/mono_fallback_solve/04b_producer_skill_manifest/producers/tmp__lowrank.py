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
rng=np.random.default_rng(82)
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(p[:52],p[52:]) for p in [rng.permutation(n) for _ in range(4)]]
X=aff(x)
# An affine input direction can modulate a vector-valued response shared by the two readouts.
def fit(i,j,kind,k,base,pen,solo=False):
 Y=T[:,:1] if solo else T; D=X if base==1 else quad(x); W=D[i]; U=np.linalg.qr(W)[0]; R=Y[i]-U@(U.T@Y[i]); d=R.std(0); R=R/d
 def features(h):
  q=X@h.reshape(k,5).T
  if kind=='gate': f=np.tanh(q)
  elif kind=='rat': f=1/(1+q*q)
  elif kind=='sin': f=np.sin(q)
  elif kind=='gauss': f=np.exp(-q*q/2)
  else: f=np.sqrt(1+q*q)
  return f if kind=='sqrt' else (f[:,:,None]*X[:,None,:]).reshape(n,-1)
 def inner(h):
  F=features(h)[i]; F=F-U@(U.T@F); P=np.linalg.solve(F.T@F+pen*np.eye(F.shape[1]),F.T@R); return F,P
 def err(h):
  F,P=inner(h); return np.r_[(R-F@P).ravel(),np.sqrt(pen)*P.ravel(),.005*h]
 best=None
 for trial in range(6):
  h=rng.normal(0,.42,(k,5));
  if trial<4: h[0,trial+1]+=1
  res=least_squares(err,h.ravel(),bounds=(-4,4),max_nfev=65,ftol=3e-5)
  if best is None or res.cost<best.cost:best=res
 h=best.x; F=features(h); P=inner(h)[1]*d; C=np.linalg.lstsq(W,Y[i]-F[i]@P,rcond=None)[0]; p=D[j]@C+F[j]@P
 return np.mean(abs(p-Y[j]),axis=0),np.mean(abs(Y[i]-D[i]@C-F[i]@P),axis=0),h.reshape(k,5)
for kind,k,base in [(f,k,b) for f in ('gate','rat','sin','gauss','sqrt') for k,b in [(1,1),(2,1),(1,2),(2,2)]]:
 for lam in (.003,.06):
  out=[fit(i,j,kind,k,base,lam) for i,j in splits[:2]]; print(kind,k,base,lam,np.round([q[0] for q in out],4).tolist(), 'train',out[0][1].round(4),'dir',out[0][2].round(2).tolist(),flush=True)
