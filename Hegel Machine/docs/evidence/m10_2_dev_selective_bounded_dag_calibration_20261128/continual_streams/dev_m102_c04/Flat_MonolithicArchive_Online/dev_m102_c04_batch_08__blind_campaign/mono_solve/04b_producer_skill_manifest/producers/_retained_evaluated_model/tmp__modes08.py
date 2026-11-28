from graph08 import *
from scipy.linalg import svd
from scipy.optimize import minimize_scalar
D=setup(s)(s['train']['x']); z,b,p,r=[D[k] for k in ('z','b','p','r')]; Y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; N=52
q=lambda a:np.c_[a[:,0]*a[:,1],a[:,0]*a[:,3],a[:,1]*a[:,2],a[:,2]*a[:,3]]
dot=lambda a,b:np.sum(a*b,axis=1)
B=aff(z,p,r); H=-np.expm1(-t[:,None]*np.array([1.8,.42]));
def score(A,J):
 w=ls(A[:N],J[:N]); E=J-A@w;return np.round(np.mean(abs(E[N:]),axis=0),5),np.round(np.std(E[:N],axis=0),5)
print('y from observed trajectory',score(aff(T),Y));print('y from selected times',[(j,score(aff(T[:,j]),Y)) for j in range(8)]); print('joint sv',svd(np.c_[T,Y][:N],compute_uv=False));print('raw T sv',svd(T[:N],compute_uv=False));
for name,A in [('base',B),('dot',np.c_[B,dot(p,r)]),('cubic',np.c_[B,np.einsum('ni,nj->nij',z,q(z)).reshape(len(z),-1)]),('cubdot',np.c_[B,np.einsum('ni,nj->nij',z,q(z)).reshape(len(z),-1),dot(p,r)]),('cubcross',np.c_[B,np.einsum('ni,nj->nij',z,q(z)).reshape(len(z),-1),np.einsum('ni,nj->nij',p,r).reshape(len(z),-1)])]:
 J=np.c_[T,Y]; E=J-A@ls(A[:N],J[:N]); _,sig,h=svd(E[:N],full_matrices=False);print(name,score(A,J),sig, np.round(h[:3],3)); print('time residual direction',np.round((H@ls(H,E[:N,:8].T)-E[:N,:8].T).std(axis=1),5))
for kind,fn in [('exp',lambda k:-np.expm1(-k*t)),('sin',lambda k:np.sin(k*t)),('cos',lambda k:1-np.cos(k*t)),('damped',lambda k:t*np.exp(-k*t)),('tanh',lambda k:np.tanh(k*t)),('rat',lambda k:t/(1+k*t)),('pulse',lambda k:t*np.exp(-k*t*t))]:
 def get(p):return np.c_[-np.expm1(-t[:,None]*p[:2]),fn(p[2])]
 def err(p):
  F=get(p);return (F@ls(F,T[:N].T)-T[:N].T).ravel()
 ans=min([least_squares(err,[1.8,.42,i],bounds=([.05,.02,.01],[8,4,12])) for i in [.22,1.25,2.4,4,6]],key=lambda r:np.linalg.norm(r.fun)); print('temporal',kind,ans.x,np.std(ans.fun),np.round(get(ans.x)[:,2],4))
