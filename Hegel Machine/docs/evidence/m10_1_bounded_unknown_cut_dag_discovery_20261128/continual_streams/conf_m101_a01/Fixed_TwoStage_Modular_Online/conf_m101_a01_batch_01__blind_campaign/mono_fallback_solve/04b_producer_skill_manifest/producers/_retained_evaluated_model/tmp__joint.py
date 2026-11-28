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
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
rng=np.random.default_rng(812); splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(p[:52],p[52:]) for p in [rng.permutation(n) for _ in range(5)]]
U={'x':x,'cal':ai(x),'mix':b(ai(x)),'inv':bi(x)}
def gp(D,K,i,j,Y):
 M=cho_factor(K[np.ix_(i,i)]); F=cho_solve(M,np.c_[D[i],Y[i]]); k=D.shape[1]; W=np.linalg.solve(D[i].T@F[:,:k],D[i].T@F[:,k:]); V=F[:,k:]-F[:,:k]@W; return D[j]@W+K[np.ix_(j,i)]@V
for name,D in [('aff',aff(x)),('quad',quad(x))]:
 R=T-D@np.linalg.lstsq(D,T,rcond=None)[0]; print('residual correlation',name,np.corrcoef(R.T)[0,1]);
for i,j in splits[:2]:
 print('oracle');
 for fm in [lambda x,z:np.c_[aff(x),z], lambda x,z:np.c_[quad(x),z],lambda x,z:np.c_[quad(x),z,z*z,z[:,None]*x],lambda x,z:np.c_[aff(x),z,z*z,z[:,None]*x],lambda x,z:np.c_[aff(x),z,z*z,z[:,None]*x,z**3,z[:,None]**2*x]]:
  W=fm(x,z); C=np.linalg.lstsq(W[i],y[i],rcond=None)[0]; print(np.mean(abs(W[j]@C-y[j])),C[-min(7,len(C)):]);
def fit_ard(V,D,Y,i,j,mode,reg):
 dd=(V[:,None,:]-V[None,:,:])**2; aa=dd[np.ix_(i,i)]; B=D[i]; Y=Y[:,None] if Y.ndim==1 else Y; R=Y[i]; m=len(i)-D.shape[1]; p=V.shape[1];
 def kern(dd,h):
  q=dd@np.exp(h[:p]);
  if mode=='sq':return np.exp(-q/2)
  if mode=='rq':return (1+q/2)**-1
  d=np.sqrt(q); return (1+d+d*d/3)*np.exp(-d)
 def loss(h):
  K=kern(aa,h)+np.exp(h[-1])*np.eye(len(i)); cf=cho_factor(K); F=cho_solve(cf,np.c_[B,R]); k=B.shape[1]; L=B.T@F[:,:k]; H=R-B@np.linalg.solve(L,B.T@F[:,k:]); C=H.T@cho_solve(cf,H); l=np.linalg.slogdet(C)[1]*m+(2*np.log(np.diag(cf[0])).sum()+np.linalg.slogdet(L)[1])*Y.shape[1]; return l+reg*np.sum((h[:p]-h[:p].mean())**2)
 h=minimize(loss,np.r_[np.full(p,-.2),-7.5],bounds=[(-5,3)]*p+[(-14,-2)],method='L-BFGS-B',options={'maxiter':85}).x
 K=kern(dd,h)+np.exp(h[-1])*np.eye(n);return gp(D,K,i,j,Y),np.round(h,2)
for name,V in U.items():
 for trname,D in [('aff',aff(x)),('quad',quad(x))]:
  for mode in ('sq','rq','m52'):
   for label,Y in [('single',y),('joint',T)]:
    e=[]
    for i,j in splits:
     P,h=fit_ard(V,D,Y,i,j,mode,1.0); e.append(np.mean(abs(P[:,0]-y[j])))
    print(name,trname,mode,label,np.round(e,4),round(np.mean(e),4),h)
