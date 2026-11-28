from graph import *
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=160)
z,A,B,C,D=ops(s,np.r_[s['train']['x'],s['test']['x']]); x=s['train']['x']; Y=s['train']['y']; T=s['train']['traj']; n=52
print('meta',s['telemetry_vec'],s['train']['summary']); print('range',z.min(0),z.max(0),Y.min(),Y.max(),'B',B(z).min(0),B(z).max(0)); print('samples',np.c_[z[:20],Y[:20],T[:20,[0,3,7]]]);print('archive');print(open('/tmp/hegel_continual_archive/conf_m102_b01_batch_00_solver.py').read())
# Profile the panel mixing matrices against shared coordinatewise transformations.
for d,key,base in [(s['panel_b'],'v','odd'),(s['panel_c'],'v2','even')]:
 u=d['u']; v=d[key];print('PANEL',base)
 fs={'odd':[('sin',lambda u,p:u+p[1]*np.sin(p[0]*u),[2.9,-.4]),('sin+cub',lambda u,p:u+p[1]*np.sin(p[0]*u)+p[2]*u**3,[2.9,-.4,0]),('sin+sin',lambda u,p:u+p[1]*np.sin(p[0]*u)+p[3]*np.sin(p[2]*u),[2.9,-.4,5.,.01]),('sin+cos',lambda u,p:u+p[1]*np.sin(p[0]*u)+p[2]*(1-np.cos(p[0]*u)),[2.9,-.4,0]),('cub',lambda u,p:u**3+p[0]*u,[.05])],'even':[('cos',lambda u,p:u+p[1]*(1-np.cos(p[0]*u)),[1.3,.65]),('cos+quad',lambda u,p:u+p[1]*(1-np.cos(p[0]*u))+p[2]*u**2,[1.3,.65,0]),('cos+sin',lambda u,p:u+p[1]*(1-np.cos(p[0]*u))+p[2]*np.sin(p[0]*u),[1.3,.65,0])]}[base]
 for name,f,p in fs:
  def resid(p):
   F=aff(f(u,p));return (F@ls(F,v)-v).ravel()
  o=least_squares(resid,p,max_nfev=300);print(name,o.x,np.std(o.fun));
# Search for a one-dimensional shared latent driver with nonlinear readout.
base={'x':np.r_[x,s['test']['x']],'z':z,'B':B(z),'C':C(z),'D':D(z),'BD':np.c_[B(z),D(z)],'zB':np.c_[z,B(z)],'zD':np.c_[z,D(z)],'BzC':np.c_[B(z),C(z)],'BC':B(z)+C(z),'CB':C(B(z)),'DB':D(B(z)),'BDcomp':B(D(z))}
q=np.c_[Y,T[:,[0,2,5,7]]]; scale=q[:n].std(0);q=q/scale
rng=np.random.default_rng(484)
for name,a in base.items():
 a=a[:72];a=(a-a[:n].mean(0))/(a[:n].std(0)+1e-9); candidates=[]
 for typ in ['poly','sin','tanh']:
  def f(w,a):
   r=a@w
   if typ=='poly':return np.c_[np.ones(len(r)),r,r*r,r**3]
   if typ=='sin':return np.c_[np.ones(len(r)),r,np.sin(r),np.cos(r)]
   return np.c_[np.ones(len(r)),r,np.tanh(r),1-np.tanh(r)**2]
  def res(w):
   F=f(w,a[:n]);return (F@ls(F,q[:n])-q[:n]).ravel()
  init=ls(aff(a[:n]),q[:n])[-a.shape[1]:,0];init=init/(np.linalg.norm(init)+1e-9); scores=[]
  for w in [init]+[rng.normal(size=len(init))*sc for sc in [.35,.7,1.3] for _ in range(6)]:
   o=least_squares(res,np.clip(w,-4.99,4.99),bounds=(-5,5),max_nfev=130); F=f(o.x,a); pred=F@ls(F[:n],q[:n]); er=abs(pred-q)*scale; scores.append((np.mean(o.fun**2),er[n:,0].mean(),er[n:,1:].mean(),o.x))
  score=min(scores,key=lambda r:r[0]); candidates.append((typ,*score))
 print('RIDGE',name,candidates,flush=True)
