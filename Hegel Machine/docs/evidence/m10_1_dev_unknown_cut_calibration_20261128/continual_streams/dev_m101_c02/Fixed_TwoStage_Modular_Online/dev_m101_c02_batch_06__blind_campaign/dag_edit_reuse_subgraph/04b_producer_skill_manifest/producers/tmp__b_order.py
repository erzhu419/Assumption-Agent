import itertools,pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
from panel_ab import fit_a,inv_a
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(p['panel_b'][k] for k in ['u','v']); t=inv_a(p['train']['x'],fit_a(p['panel_a'])); Y=p['train']['traj']; print('ranges A inverse',t.min(0),t.max(0),'B input',u.min(0),u.max(0))
from runpy import run_path
sink=run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
tr=np.arange(52); va=np.arange(52,72)
pairs=list(itertools.combinations_with_replacement(range(4),2))
def base(x): return np.c_[np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in pairs]]
def score(z):
 A=np.c_[np.ones(len(z)),z]; E=A[va]@np.linalg.lstsq(A[tr],Y[tr],rcond=None)[0]-Y[va]; return np.mean(abs(E))
def report(name,predict):
 z=predict(t); print(name,'B fit',np.sqrt(np.mean((predict(u)-v)**2)),'traj CV',score(z),flush=True)
for use_cos in (False,True):
 def mod(x,f):
  X=np.c_[base(x),np.sin(x*f)]
  if use_cos:X=np.c_[X,np.cos(x*f)]
  return X
 def proj(f):
  X=mod(u,f); c=np.linalg.lstsq(X,v,rcond=None)[0]; return X@c-v,c
 f=minimize_scalar(lambda f:np.linalg.norm(proj(f)[0]),bounds=(1.,4.),method='bounded').x; e,c=proj(f); report(('generic',use_cos,f),lambda x:mod(x,f)@c)
 if not use_cos: print('linear\n',c[1:5],'sin\n',c[-4:]); print('ratios',c[-4:]/c[1:5])
# Fit directional sine with the same quadratic nuisance basis, testing isolated axis, paired mixing, and full direction.
B=base(u); Q=np.linalg.qr(B)[0]; R=lambda a:a-Q@(Q.T@a); rv=R(v)
A=np.linalg.lstsq(np.c_[B,np.sin(2.85*u)],v,rcond=None)[0][1:5]; A/=np.linalg.norm(A,axis=0)
maskpair=np.array([[1,1,0,0],[0,0,1,1],[0,0,1,1],[1,1,0,0]],bool)[:,[0,2,3,1]]
for initial,mask in [(np.eye(4),np.eye(4,dtype=bool)),(np.eye(4),maskpair),(np.eye(4),np.ones((4,4),bool)),(A,np.ones((4,4),bool))]:
 def unpack(w):
  D=np.zeros((4,4)); D[mask]=w; return D
 def fun(w):
  S=R(np.sin(u@unpack(w))); return (S@np.linalg.lstsq(S,rv,rcond=None)[0]-rv).ravel()
 res=least_squares(fun,(initial*2.85)[mask],max_nfev=1000,ftol=1e-11,gtol=1e-11); D=unpack(res.x); C=np.linalg.lstsq(np.c_[B,np.sin(u@D)],v,rcond=None)[0]; report(('directions',np.round(D,3)),lambda x:np.c_[base(x),np.sin(x@D)]@C); print('linear quad sine\n',np.round(C,4))
