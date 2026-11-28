from h import *
from scipy.optimize import least_squares
rng=np.random.default_rng(301)
R=np.linalg.svd(T[:n],full_matrices=False)[2][:3]; yy=np.c_[y,T@R.T]; sd=yy[:n].std(0); V=yy[:n]/sd; V[:,2:]*=.58; sd[2:]/=.58
stuff={'q':q,'z':z,'C':C(q),'D':D(q),'CB':C(z),'DB':D(z),'qCD':np.c_[q,C(q),D(q)],'CD':np.c_[C(q),D(q)],'CBD':np.c_[C(z),D(z)]}
def report(f):
 U,S,W=np.linalg.svd(f[:n],full_matrices=False); m=S>1e-8*S[0]; U,S,W=U[:,m],S[m],W[m]; E=U@(U.T@V)-V; H=f[n:]@(W.T@((U.T@V)/S[:,None])); L=E/(1-(U*U).sum(1))[:,None]*sd; H=H*sd-yy[n:];return np.array([np.abs(L[:,0]).mean(),np.abs(L[:,1:]@R).mean(),np.abs(H[:,0]).mean(),np.abs(H[:,1:]@R).mean()])
results=[]
for key,p in stuff.items():
 p=(p-p[:n].mean(0))/p[:n].std(0); dim=p.shape[1]
 for basekey,f in [('lin',A(q)),('quad',poly(q,2)),('pan',A(np.c_[q,C(q),D(q)]))]:
  if dim>4 and basekey=='quad':continue
  uf,sf,wf=np.linalg.svd(f[:n],full_matrices=False); uf=uf[:,sf>1e-8*sf[0]]; f=f@wf[sf>1e-8*sf[0]].T
  def resid(v):return v-uf@(uf.T@v)
  W=resid(V)
  for act in ('cos','tanh','rat','gauss'):
   def fun(a,p0):
    h=p0@a[:,:dim].T+a[:,dim];
    if act=='cos':return np.cos(h)
    if act=='tanh':return np.tanh(h)
    if act=='rat':return 1/(1+h*h)
    return np.exp(-h*h)
   def opt(a):
    h=resid(fun(a.reshape(-1,dim+1),p[:n])); c=ls(h,W);return (h@c-W).ravel()
   best=None
   for _ in range(6):
    a=rng.normal(size=(2,dim+1))*.8
    o=least_squares(opt,a.ravel(),max_nfev=85,bounds=(-7,7)); e=np.linalg.norm(o.fun)
    if best is None or e<best[0]:best=(e,o.x.reshape(2,-1))
   a=best[1]; F=np.c_[f,fun(a,p)]; m=report(F); results.append((best[0],key,basekey,act,m,a));
for e,k,b0,a,m,v in sorted(results,key=lambda o:o[0])[:16]:print(round(e,4),k,b0,a,m,np.round(v,3))
