exec(open('/tmp/link_probe.py').read().split("for key in ('v1','v2')")[0])
u=c['u']; v=c['v1'];
def DD(q,kind,k=1):
 if kind=='odd3': h=q**3
 if kind=='abs2': h=q*abs(q)
 if kind=='rat': h=q*abs(q)/(1+k*abs(q))
 if kind=='rat2': h=q**3/(1+k*q*q)
 if kind=='tanh': h=np.tanh(k*q)
 if kind=='sin': h=np.sin(k*q)
 if kind=='asinh': h=np.arcsinh(k*q)
 if kind=='cube2': return np.c_[np.ones(len(q)),q,q*q,q**3]
 if kind=='cube4': return np.c_[np.ones(len(q)),q,q*q,q**3,q**4]
 return np.c_[np.ones(len(q)),q,h]
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
for kind in ['odd3','abs2','rat','rat2','tanh','sin','asinh','cube2','cube4']:
 k=1.
 if kind in ('rat','rat2','tanh','sin','asinh'):
  k=np.exp(minimize_scalar(lambda p: np.linalg.norm((D:=DD(u,kind,np.exp(p)))@np.linalg.lstsq(D,v,rcond=None)[0]-v),bounds=(-5,3),method='bounded').x)
 D=DD(u,kind,k); A=np.linalg.lstsq(D,v,rcond=None)[0]; w=DD(z,kind,k)@A
 print(kind,k,'panel rms',np.mean((D@A-v)**2)**.5,'sv',np.linalg.svd(A[5:],compute_uv=False))
 if kind in ('odd3','cube2'): print(np.round(A,4))
 for token in ('all','diag','Bdiag'):
  q=x if token!='Bdiag' else z
  inter=(q[:,:,None]*w[:,None,:]).reshape(len(q),-1) if token=='all' else q*w
  H=np.c_[np.ones(len(q)),q,w,inter]; choose(H,regs,kind+' '+token)
 print(flush=True)
