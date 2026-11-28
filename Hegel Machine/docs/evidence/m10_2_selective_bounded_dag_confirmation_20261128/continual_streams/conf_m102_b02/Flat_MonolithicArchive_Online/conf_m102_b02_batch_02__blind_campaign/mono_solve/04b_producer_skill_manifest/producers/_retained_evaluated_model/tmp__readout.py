from explore import *
np.set_printoptions(precision=5,suppress=True,linewidth=140)
P=np.c_[np.ones(len(w)),w,powr(w,2),powr(w,3)]; print('canonical coefficients (y, trajectory at first and last horizons)'); print(ls(P,np.c_[y,s['train']['traj'][:,[0,-1]]]))
q=aug(powr(w,2))@C; q-=C[0]
groups={'w':w,'q':q,'wq':w*q,'wq_full':(w[:,:,None]*q[:,None,:]).reshape(-1,16),'cube':powr(w,3),'w_cube':w**3,'wquad':w*w,'c2':c2fun(w),'z':z,'zwq':z*q}
specs=[('w','q'),('w','q','wq'),('w','q','wq_full'),('w','q','cube'),('w','q','w_cube'),('w','q','wq','w_cube'),('w','q','wq','wquad'),('w','q','wq','c2'),('w','q','zwq'),('w','q','cube','wquad'),('w','q','cube','c2')]
Y=np.c_[y,s['train']['traj']]; rng=np.random.default_rng(492); ind=np.arange(72); splits=[(ind[:52],ind[52:])]
for _ in range(100):
 i=rng.permutation(ind); splits.append((i[:52],i[52:]))
for sp in specs:
 F=aug(np.column_stack([groups[key] for key in sp])); E=[]
 for a,b in splits:
  T=Y[a,1:]; modes=np.linalg.svd(T,full_matrices=False)[2][:2]; Yp=F[b]@ls(F[a],np.c_[Y[a,0],T@modes.T]); err=np.abs(np.c_[Yp[:,0],Yp[:,1:]@modes]-Y[b]); E.append([err[:,0].mean(),err[:,1:].mean(),err[:,1:].mean(0).max()]);
 print(sp,np.linalg.matrix_rank(F),'fixed',E[0],'median',np.median(E,axis=0),'worst',np.max(E,axis=0));
 if sp==('w','q','wq'): print('structured coefficients',ls(F,np.c_[y,Y[:,[1,-1]]]))
# Test output link hypotheses for C2 on the parsimonious shared feature u + k*u**2.
from scipy.special import erf
for name,fn in [('linear',lambda a:a),('tanh',np.tanh),('sin',np.sin),('erf',erf),('asinh',np.arcsinh),('atan',np.arctan)]:
 E=[]; PP=[]
 for j in range(4):
  v=v2[:,j]; L=ls(aug(u+.45*u*u),v)
  def pred(p,uu): return p[0]+np.exp(p[1])*fn(p[2]+(uu+p[3]*uu*uu)@p[4:]);
  p=np.r_[0.,0.,L[0],.45,L[1:]]; f=least_squares(lambda p:pred(p,u)-v,p,max_nfev=500,bounds=([-3,-3,-3,-2]+[-4]*4,[3,3,3,2]+[4]*4)); E.append(np.sqrt(np.mean(f.fun**2)));PP.append(f.x[:4]);
 print('C2',name,E,PP)
