import numpy as np,pickle,runpy
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
p='/holdout_task/'; s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s}); b=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']({'node_00':a}); u,z=a['u_tr'],b['z_tr']; y=s['train']['y']; n=len(y); ix=np.arange(n)
poly=lambda x,d: np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
L=poly(u,1); Q=poly(u,2); T=poly(z,2); C=s['panel_c']; D=np.linalg.lstsq(poly(C['u'],2),C['v2'],rcond=None)[0]; print('C v2 quadratic coefficients',D.round(3)); H=Q@D
splits=[ix[52:]]+list(np.split(np.random.default_rng(982).permutation(n),6))+[np.argsort(np.linalg.norm(u,axis=1))[-12:]]
for name,A in [('affine',L),('C-v2',np.c_[L,H]),('quadratic',Q)]:
 for kind in ('tanh','atan','erf','softroot'):
  from scipy.special import erf
  f={'tanh':np.tanh,'atan':np.arctan,'erf':erf,'softroot':lambda v:v/np.sqrt(1+v*v)}[kind]
  for skip in (False,True):
   def fit(tr,te):
    q=0 if not skip else L.shape[1]; initial=np.r_[np.zeros(q),2.,np.linalg.lstsq(A[tr],y[tr]/2,rcond=None)[0]]
    def pred(v,j): return (L[j]@v[:q] if skip else 0)+v[q]*f(A[j]@v[q+1:]);
    def obj(v): return np.r_[pred(v,tr)-y[tr],.003*(v-initial)]
    r=least_squares(obj,initial,max_nfev=160); return np.mean(abs(pred(r.x,te)-y[te])),r.x
   mm=np.array([fit(np.setdiff1d(ix,te),te)[0] for te in splits]); full,zz=fit(ix,ix); print(name,kind,skip,np.round([mm[0],mm[1:7].max(),mm[7],full,zz[5 if skip else 0]],5))
# Pair-specific smooth ridge terms: tanh of u_i +/- z_j, and one-coordinate saturation times the other coordinate.
from scipy.linalg import qr
for typ in ('tanh','sin','atan','rat'):
 f={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rat':lambda v:v/(1+v*v)}[typ]; X=np.c_[u,z]; combos=[(i,j,c) for i in range(4) for j in range(4,8) for c in (-2,-1,-.5,.5,1,2)]; V=np.column_stack([X[:,i]+c*X[:,j] for i,j,c in combos]); xx=np.c_[X,V]; models=[]
 for scale in (.5,1,1.7,2,3):
  F=f(scale*xx); G=f(scale*X); B=np.c_[F,(X[:,:,None]*G[:,None,:]).reshape(n,-1)]; B/=np.sqrt(B.shape[1]); K=B@B.T; P=(1+u@u.T)**3/25.;
  for w in (.01,.1,1.):
   for ridge in (.0001,.001,.01):
    out=[]
    for te in splits:
     tr=np.setdiff1d(ix,te); J=K+w*P; M=J[np.ix_(tr,tr)]+ridge*np.eye(len(tr)); Vv=np.linalg.solve(M,np.c_[L[tr],y[tr]]); beta=np.linalg.solve(L[tr].T@Vv[:,:5],L[tr].T@Vv[:,5]); out.append(np.mean(abs(L[te]@beta+J[np.ix_(te,tr)]@(Vv[:,5]-Vv[:,:5]@beta)-y[te])))
    models.append((max(out)+.2*np.mean(out),scale,w,ridge,np.round([out[0],max(out[1:7]),out[-1]],5)))
 print(typ,sorted(models,key=lambda t:t[0])[:3])
