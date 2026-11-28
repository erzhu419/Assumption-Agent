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
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
rng=np.random.default_rng(812); splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(p[:52],p[52:]) for p in [rng.permutation(n) for _ in range(8)]]
D=aff(x); Q=quad(x)[:,5:]; H=np.c_[*[np.prod(x[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),3)]]

def model(i,j,Y,kind,reg,extras=0,mean='aff'):
 V=bi(x); S=np.c_[V,x]; dd=V[:,None,:]-V[None,:,:]; ix=np.ix_(i,i); di=dd[ix]; ij=np.tril_indices(4,-1); off=np.zeros((4,4)); D0=D if mean=='aff' else np.c_[D,Q]; W=D0[i]; Y=Y.reshape(n,-1); R=Y[i]; m=len(i)-W.shape[1]; ny=R.shape[1]
 L0=np.eye(4); P=Q@Q.T; C=H@H.T; extra=np.array([P,C]); E=extra[:,i[:,None],i]; G=np.sum(di**2,axis=2)
 def kernel(d,h,e):
  L=np.diag(np.exp(h[:4]/2)); L[ij]=h[4:10]; q=np.sum((d@L)**2,axis=2)
  if kind=='sq': K=np.exp(-q/2)
  elif kind=='rq': K=1/(1+q/2)
  else:
   t=np.sqrt(q); K=(1+t+t*t/3)*np.exp(-t)
  if extras: K=K+np.einsum('k,kij->ij',np.exp(h[11:]),e[:extras]);
  return K
 def objective(h):
  K=kernel(di,h,E)+np.exp(h[10])*np.eye(len(i)); cf=cho_factor(K); F=cho_solve(cf,np.c_[W,R]); p=W.shape[1]; A=W.T@F[:,:p]; b=W.T@F[:,p:]; U=R-W@np.linalg.solve(A,b); C=U.T@cho_solve(cf,U); cost=m*np.linalg.slogdet(C)[1]+ny*(2*np.log(np.diag(cf[0])).sum()+np.linalg.slogdet(A)[1]); return cost+reg*(np.dot(h[4:10],h[4:10])*4+np.var(h[:4])*2)
 h=minimize(objective,np.r_[np.full(4,-1.),np.zeros(6),-10.,np.full(extras,-4.)],bounds=[(-6,3)]*4+ [(-1.5,1.5)]*6+[(-16,-2)]+[(-12,3)]*extras,method='L-BFGS-B',options={'maxiter':105}).x
 K=kernel(dd,h,extra); cf=cho_factor(K[ix]+np.exp(h[10])*np.eye(len(i))); F=cho_solve(cf,np.c_[W,R]); p=W.shape[1]; C=np.linalg.solve(W.T@F[:,:p],W.T@F[:,p:]); return D0[j]@C+K[np.ix_(j,i)]@(F[:,p:]-F[:,:p]@C)

if __name__=='__main__':
 out=[]
 for kind in ['sq','rq','m52']:
  for ytype,Y in [('joint',T),('solo',y)]:
   for reg,ex,mean in [(1,0,'aff'),(8,0,'aff'),(30,0,'aff'),(8,1,'aff'),(8,2,'aff'),(8,0,'quad')]:
    pred=[model(i,j,Y,kind,reg,ex,mean) for i,j in splits]; err=np.array([np.mean(abs(p[:,0]-y[j])) for p,(_,j) in zip(pred,splits)]); out.append((err.mean(),(kind,ytype,reg,ex,mean),pred));print(out[-1][1],err.round(4),err.mean().round(4),flush=True)
 for m in [3,6,12,len(out)]:
  best=sorted(out,key=lambda o:o[0])[:m]; err=np.array([np.mean(abs(np.mean([o[2][k][:,0] for o in best],axis=0)-y[j])) for k,(_,j) in enumerate(splits)]); print('ensemble',m,err.round(4),err.mean().round(4),err.max().round(4))
