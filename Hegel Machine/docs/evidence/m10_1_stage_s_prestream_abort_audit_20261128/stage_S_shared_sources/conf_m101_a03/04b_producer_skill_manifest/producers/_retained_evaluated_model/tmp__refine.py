from structure import *
from itertools import combinations_with_replacement
w=(x-M[0])/np.diag(M[1:]); u=aff(w)@N
D=aff(u); G=np.c_[u[:,0]*u[:,2],u[:,1]*u[:,3]]; Q=quad(u)
models={'shared_pair':np.c_[D,G.sum(1)],'distinct_pairs':np.c_[D,G],'full_quad':Q,'pair_no_intercept':np.c_[u,G.sum(1)],'pair_fixed_018':D}
print('scalar fits')
for k,F in models.items():
 yy=y-.18*G.sum(1) if k=='pair_fixed_018' else y
 coef=ls(F,yy); print(k,coef, 'rms',np.sqrt(np.mean((yy-F@coef)**2)))
print('trajectory coefficients raw/calibrated/composed')
for q in (x,w,u):
 d=aff(q); C=ls(d,z); print(C)
 def basis(p): return -np.expm1(-t[:,None]*p)
 def err(p):
  T=basis(p); return (C-ls(T,C.T).T@T.T).ravel()
 opt=least_squares(err,[.42,1.82],bounds=([.02,.9],[.9,4])); T=basis(opt.x); print('rate/amplitudes/residual',opt.x,ls(T,C.T).T,err(opt.x).reshape(C.shape),sep='\n')
print('cross-validation')
for label,folds in [('repeat-random',[f for seed in (3,91,201) for f in np.array_split(np.random.default_rng(seed).permutation(len(x)),4)]),('tail',[np.arange(56,len(x))]),('directional extremes',[np.argsort(q)[-12:] for q in [*w.T,*(-w).T,*u.T,*(-u).T]])]:
 res=[]
 for va in folds:
  tr=np.setdiff1d(np.arange(len(x)),va); row=[]
  for k,F in models.items():
   yy=y-.18*G.sum(1) if k=='pair_fixed_018' else y
   row.append(np.mean(abs(yy[va]-F[va]@ls(F[tr],yy[tr]))))
  A=aff(w); X=A[tr]; Z=z[tr]; C=ls(X,Z); P=A[va]@C
  def pred(p):
   T=basis(p); return ls(T,C.T).T@T.T
  opt=least_squares(lambda p:(X@(C-pred(p))).ravel(),[.42,1.82],bounds=([.02,.9],[.9,4])); S=A[va]@pred(opt.x)
  U,_,V=np.linalg.svd(X@C,full_matrices=False); V=V[:2]; R=P@V.T@V
  zs=[P,S,R,A[va]@pred([.42,1.8])]
  row.extend([np.mean(abs(v-z[va])) for v in zs]); row.extend([np.mean(abs(v-z[va]),axis=0).max() for v in zs]);res.append(row)
 print(label,list(models),'affine / two-exp / rank-2 / fixed-exp respectively (averaged horizons then max per-horizon)');print('mean',np.mean(res,0));print('worst',np.max(res,0))
