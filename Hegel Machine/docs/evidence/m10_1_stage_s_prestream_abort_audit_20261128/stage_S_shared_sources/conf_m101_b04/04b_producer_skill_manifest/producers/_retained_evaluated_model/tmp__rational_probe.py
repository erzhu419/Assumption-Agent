from probe import *
from scipy.optimize import least_squares
from scipy.linalg import polar
x=s['train']['x']; Z=s['train']['traj']; y=s['train']['y']; I=np.arange(len(x)); folds=[I[56:]]+list(np.array_split(np.random.default_rng(42).permutation(len(I)),4))+[np.argsort(np.linalg.norm(x,axis=1))[-12:]]
F={d:poly(x,d) for d in [1,2,3]}
p=s['panel_b']; P=fit(poly(p['u'],1),p['v']); print('Panel B affine residual',np.std(poly(p['u'],1)@P-p['v']), 'singular values',np.linalg.svd(P[1:],compute_uv=False),'offset',P[0]); print('times',s['eval_times'])

def predict(t,v,head,degree,form,reg):
 Y=y[:,None] if head==0 else Z; sub=np.eye(Y.shape[1]) if head==0 else np.linalg.svd(Y[t],full_matrices=False)[2][:2].T; T=Y[t]@sub
 X=F[degree]; Q=F[2][:,1:]; A=X[t]; lam=np.zeros(A.shape[1]); lam[15:]=reg
 def den(D,p):
  if form=='quad':return np.exp(np.clip(D@p,-4,4))
  if form=='rank1':return 1+(p[0]+D[:,:4]@p[1:])**2
 def inner(p):
  N=A/den(Q[t],p)[:,None]; B=np.linalg.solve(N.T@N+np.diag(lam+1e-10),N.T@T); return N,B
 def res(p):
  N,B=inner(p); return np.r_[(N@B-T).ravel(),(lam[:,None]**.5*B).ravel(), .02*p]
 rng=np.random.default_rng(57); starts=[np.zeros(14)] if form=='quad' else [rng.normal(0,.3,5) for _ in range(5)]; r=min((least_squares(res,p,max_nfev=110) for p in starts),key=lambda a:a.cost); N,B=inner(r.x)
 return (X[v]@B/den(Q[v],r.x)[:,None])@sub.T, np.sqrt(np.mean((N@B-T)**2)),r.x
for head in (0,1):
 for degree in (2,3):
  for form in ('quad','rank1'):
   for reg in ([0] if degree==2 else [.002,.03]):
    L=[predict(np.setdiff1d(I,v),v,head,degree,form,reg) for v in folds]; Y=y[:,None] if head==0 else Z; E=np.array([np.mean(abs(P-Y[v]),axis=0) for v,(P,_,_) in zip(folds,L)]); print(head,degree,form,reg,'avg/worst/edge',E[:-1].mean(),E[:-1].max(),E[-1].max(),'fit',np.mean([r[1] for r in L]),'p first',L[0][2]);
