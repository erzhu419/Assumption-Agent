from sparse_heads import *
from scipy.linalg import solve
from scipy.optimize import least_squares
tr=s['train']; z,b,c,d=coords(s,tr['x']); z,b=old.parts(s); z=z[:72]; b=b[:72]; f=old.times(s); T=tr['traj']; y=tr['y']; shape=np.linalg.svd(T,full_matrices=False)[2][0]; target=np.c_[y,(T@shape)*shape[-1]]; rng=np.random.default_rng(31); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in (rng.permutation(72) for _ in range(3))];
for a,i in splits:
 X=np.c_[z,b]; R=np.column_stack([z[:,j]*z[:,(j+1)%4] for j in range(4)]); S=np.column_stack([b[:,j]*b[:,(j+1)%4] for j in range(4)]); Y=target; grades=[]
 for name,H in [('zb',X),('zbc',np.c_[X,c]),('zbd',np.c_[X,d]),('zbRS',np.c_[X,R,S])]:
  H=H/np.maximum(H[a].std(0),.05); P=D(H); Pi=np.linalg.pinv(P[a]); res=Y[a]-P[a]@(Pi@Y[a]);
  for typ,fun in [('tanh',np.tanh),('sin',np.sin),('rat',lambda v:v/(1+v*v))]:
   for m in [1,2,3,4]:
    outs=[]
    for k in range(2):
     best=None; yy=Y[:,k]; ini=Pi@yy[a];
     def make(p,ii):
      w=p.reshape(m,-1); return P[ii]@ini+fun(P[ii]@w[:,:-1].T)@w[:,-1]
     for j in range(5):
      init=rng.normal(0,.3,(m,P.shape[1]+1)); init[:,-1]=.5
      if j==0:init[0,:-1]=ini*.7
      def err(p):
       raw=make(p,a)-yy[a]; return np.r_[raw,.045*p]
      sol=least_squares(err,init.ravel(),max_nfev=100,ftol=.0001)
      if best is None or sol.cost<best.cost:best=sol
     outs.append((np.mean(abs(make(best.x,i)-yy[i])),np.sqrt(2*best.cost/52)))
    grades.append((name,typ,m,np.round(outs,4)))
 print('FIT/VAL',[(n,t,m,g.tolist()) for n,t,m,g in sorted(grades,key=lambda q:q[3][:,1].sum())[:8]],flush=True);print('VAL',[(n,t,m,g.tolist()) for n,t,m,g in sorted(grades,key=lambda q:q[3][:,0].sum())[:8]],flush=True)
