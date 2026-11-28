from ops import *
from scipy.special import erf
from itertools import combinations_with_replacement as cr
from pathlib import Path
T=Y[:,1:]; t=s['eval_times']; V=np.linalg.svd(T[:52],full_matrices=False)[2][:2]
print('SVD',np.linalg.svd(T,compute_uv=False))
B=np.c_[-np.expm1(-1.813*t),t/(1+.405*t)]; Z=np.c_[Y[:,0],T@np.linalg.pinv(B.T)]
D=np.load('/tmp/nodes.npz'); v=s['panel_b']['v'];u=s['panel_b']['u']; _,k,w=fit(u,v,'erf',True); oldB=lambda x:design(x,'erf',k,True)@w; v=s['panel_c']['v1'];u=s['panel_c']['u'];_,k,w=fit(u,v,'asinh',True); oldC=lambda x:design(x,'asinh',k,True)@w
rng=np.random.default_rng(417); splits=[np.arange(72)]+[rng.permutation(72) for j in range(24)]; splits += [np.argsort(x[:,j]*sgn) for j in range(4) for sgn in [-1,1]]
for name,A in [('old',np.c_[x,oldC(oldB(x))]),('refined',np.c_[x,D['bc']]),('refB',np.c_[x,oldC(D['b'])]),('refC',np.c_[x,D['bc']]),('with_b',np.c_[x,D['bc'],D['b']]),('with_bd',np.c_[x,D['bc'],D['bd']])]:
 for variant in ['full','cross','self','linear']:
  a=A/A.std(0); pairs=list(cr(range(a.shape[1]),2)); inds=[(j,k) for j,k in pairs if variant=='full' or (variant=='cross' and (j==k or j//4!=k//4)) or (variant=='self' and j//4==k//4)]; X=np.c_[np.ones(len(a)),a,*[a[:,j]*a[:,k] for j,k in inds]]
  for reg in [2e-6,.001,.01]:
   errors=[]
   for sidx,p in enumerate(splits):
    it,iv=p[:52],p[52:]; XX=X[it]; c=np.linalg.solve(XX.T@XX+reg*np.eye(X.shape[1]),XX.T@Z[it]); e=X[iv]@c-Z[iv]; errors.append([abs(e[:,0]).mean(),abs(e[:,1:]@B.T).mean()]);
    if sidx==0 and variant=='full' and reg==.001 and name in ['old','refined']:
     print('coefs',name,np.round(c[:9],2)); print([(j,k,*np.round(v,2)) for (j,k),v in zip(inds,c[9:]) if max(abs(v))>.08]);print('train mae',np.mean(abs(XX@c-Z[it]),0))
   e=np.asarray(errors); print(name,variant,reg,e[0].round(5),e[1:25].mean(0).round(5),e[1:25].max(0).round(5),e[25:].max(0).round(5))
