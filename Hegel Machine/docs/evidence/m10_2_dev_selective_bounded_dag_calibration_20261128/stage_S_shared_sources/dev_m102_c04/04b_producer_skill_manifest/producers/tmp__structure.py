import sys,pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task');import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x,y,z=[s['train'][k] for k in ['x','y','traj']];t=s['eval_times'];a,w=m._calibration(s['panel_a']);q=np.linalg.solve(w.T,(x-a).T).T;p,r=[f(q) for f in m._panel_operators(s['panel_c'])]; np.set_printoptions(precision=4,suppress=True)
u=s['panel_c']['u']; B=m._ls(m._affine(s['panel_b']['u']),s['panel_b']['v']);print('cal',a,w,'B',B,'telemetry',s['telemetry_vec']);
for k in [1,2]:
 v=s['panel_c']['v'+str(k)]; K=.8271 if k==1 else .16388; h=u/(1+K*abs(u)) if k==1 else u*u/(1+K*u*u); W=m._ls(np.c_[m._affine(u),h],v);print('panel',k,W)
V=-np.expm1(-t[:,None]*[1.805,.427]); z2=m._ls(V,z.T).T; print('svd',np.linalg.svd(z,compute_uv=False));print('amp readout weights',m._ls(m._affine(np.c_[p,r]),np.c_[y,z2])); print('baseline feature weights',m._ls(m._affine(m._features(p,r)),np.c_[y,z2]).T)
fold=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(671)
for _ in range(20):
 a=rng.permutation(72);fold.append((a[:52],a[52:]))
for b in [*x.T,np.linalg.norm(x,axis=1)]:
 for d in [-1,1]:
  a=np.argsort(d*b);fold.append((a[:52],a[52:]))
yz=np.c_[y,z]; scores=[]
def check(tag,F,regs=(1e-7,1e-5,1e-4,1e-3,.01)):
 if np.min(F.std(0))<1e-12:F=F[:,F.std(0)>1e-12]
 for lam in regs:
  E=[]
  for i,j in fold:
   ff=F-F[i].mean(0);Y=yz[i]-yz[i].mean(0); wi=np.linalg.solve(ff[i].T@ff[i]+lam*np.eye(F.shape[1]),ff[i].T@Y);e=abs(ff[j]@wi+yz[i].mean(0)-yz[j]);E.append([e[:,0].mean(),e[:,1:].mean()])
  E=np.array(E); scores.append((tag,lam,F.shape[1],E[0],E[1:21].max(0),E[21:].max(0),E[21:].mean(0)))
base=m._features(p,r);prod=(p[:,:,None]*r[:,None,:]).reshape(-1,16);dprod=p*r
check('base',base)
for name,F in [('pr',np.c_[p,r]),('pr+diag',np.c_[p,r,dprod]),('pr+outer',np.c_[p,r,prod]),('pr+square',np.c_[p,r,p*p,r*r,prod])]:
 for g,G in [('none',np.empty((72,0))),('tanh',np.c_[np.tanh(p),np.tanh(r)]),('sin',np.c_[np.sin(p),np.sin(r)]),('squares',np.c_[p*p,r*r]),('rat',np.c_[p/(1+p*p),r/(1+r*r)]),('q',q),('soft',np.c_[p/(1+abs(p)),r/(1+abs(r))]),('diff',np.c_[np.tanh(p-r),np.tanh(p+r)]),('prod nonlinear',np.c_[np.tanh(dprod),np.sin(dprod)])]:check(name+' '+g,np.c_[F,G])
for i in [0,1]:
 print('BEST tail',i);[print(k) for k in sorted(scores,key=lambda a:a[5][i])[:12]]
print('base values');[print(k) for k in scores if k[0]=='base']
