import sys;sys.path.insert(0,'/holdout_task');import batch_solver as b
from structure import *
from scipy.optimize import least_squares
def fit(d,k,fun,starts,bounds):
 u,v=s[d]['u'],s[d][k]
 def res(p):
  A=aff(fun(u,p)); return (A@ls(A,v)-v).ravel()
 o=min((least_squares(res,p,bounds=bounds,max_nfev=200) for p in starts),key=lambda r:np.linalg.norm(r.fun));m=ls(aff(fun(u,o.x)),v);print(d,k,o.x,np.std(o.fun));return lambda x:aff(fun(x,o.x))@m
B=[]
for fn,starts,bounds in [(lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[[.35,1.5]],([0,.25],[.8,4])),(lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**1.5,[[.35]],([0],[.8])),(lambda x,p:np.sign(x)*np.maximum(abs(x)-.35,0)**1.5,[[.35]],([0],[.8]))]:
 B.append(fit('panel_b','v',fn,starts,bounds))
C=[]
for fn,p,lo,hi in [(lambda x,p:x/(1+p[0]*abs(x))**p[1]+p[2]*x,[.65,1.,.1],[.01,.1,-2],[5,5,2]),(lambda x,p:x/(1+p[0]*abs(x))**p[1],[.65,1.],[.01,.1],[5,5]),(lambda x,p:x/(1+p[0]*abs(x)+p[1]*x*x+p[2]*abs(x)**3)+p[3]*x,[.98,.37,.05,.087],[0,0,0,-1],[20,20,20,1]),(lambda x,p:np.sign(x)*np.log1p(p[0]*abs(x))+p[1]*x,[1.,0.],[.01,-5],[10,5]),(lambda x,p:(np.sign(x)*np.log1p(p[0]*abs(x)))+p[1]*x*x*x+p[2]*x,[1,0,0],[.01,-5,-5],[10,5,5])]:
 C.append(fit('panel_c','v1',fn,[p],(lo,hi)))
X=b._calibrate(s['panel_a'],s['train']['x']); Y=s['train']['y'];T=s['train']['traj']; A=np.arange(len(X)); folds=[(A[:52],A[52:])]; rng=np.random.default_rng(517)
for i in range(16):
 j=rng.permutation(A);folds.append((j[:52],j[52:]));
for j in np.argsort(abs(X),axis=0).T:
 folds.append((j[:-20],j[-20:]));
for i,f in enumerate(B):
 for j,g in enumerate(C):
  z=g(f(X)); D=b._scalar_design(z); E=aff(z); stats=[]
  for tr,te in folds:
   dy=np.abs(D[te]@ls(D[tr],Y[tr])-Y[te]); dt=np.abs(E[te]@ls(E[tr],T[tr])-T[te]); stats.append([dy.mean(),dt.mean(),dt.mean(0).max(),dy.max(),dt.max()]);
  print(i,j,'52/20',np.round(stats[0],6),'worst',np.round(np.max(stats,axis=0),6),'rank',np.linalg.svd(T-E@ls(E,T),compute_uv=False)[:2]);
