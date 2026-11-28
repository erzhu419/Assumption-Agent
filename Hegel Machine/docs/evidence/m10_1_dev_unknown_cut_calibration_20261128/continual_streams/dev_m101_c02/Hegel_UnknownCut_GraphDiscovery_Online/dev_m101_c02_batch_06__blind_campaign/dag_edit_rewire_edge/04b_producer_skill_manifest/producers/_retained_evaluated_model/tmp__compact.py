from explore import *
from itertools import combinations
F=dic(u,bv); main=list(F)[:9]; cyc=[f'u{i}*b{(i-1)%4}' for i in range(4)]; cubic=[f'u{i}^3' for i in range(4)]; pairs=[f'u{i}*u{j}' for i in range(4) for j in range(i,4)]; cross=[f'u{i}*b{j}' for i in range(4) for j in range(4)]
rng=np.random.default_rng(29); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for i in range(12)]]; Z=np.column_stack((y,tr)); print('metrics: public y/traj, worst y/traj, fit y/traj')
for name, cols in [('main',main),('cyclic',main+cyc),('cyclic+cubic',main+cyc+cubic),('cyclic+u quadratic',main+cyc+pairs),('cyclic+u polynomial',main+cyc+pairs+cubic),('all cross',main+cross),('cross+cubic',main+cross+cubic),('u and cyclic',[*main[:5],*cyc]),('B and cyclic',[main[0],*main[5:],*cyc])]:
 X=np.column_stack([F[k] for k in cols]); pp=[]
 for a0,a1 in splits:
  e=abs(X[a1]@np.linalg.lstsq(X[a0],Z[a0],rcond=None)[0]-Z[a1]); pp.append([e[:,0].mean(),e[:,1:].mean()]);
 w=np.linalg.lstsq(X,Z,rcond=None)[0]; e=abs(X@w-Z); print(name,len(cols),np.round(pp[0],6),np.round(np.max(pp,0),6),np.round([e[:,0].mean(),e[:,1:].mean()],6));
 if name in ['cyclic','cyclic+cubic']:
  print('coefficients:',[(col,*np.round(c[[0,1,4,8]],4)) for col,c in zip(cols,w)])
  print('residual OMP y',omp(F,Z[:,0]-X@w[:,0],8)); print('temporal coeff sv',np.linalg.svd(w[:,1:],compute_uv=False)); print('resid sv',np.linalg.svd((X@w-Z)[:,1:],compute_uv=False))
  print('temporal coeffs',np.round(w[:,1:],3))
# Test the panel-B harmonic-plus-quadratic grammar independently of response labels.
P=s['panel_b']['u']; V=s['panel_b']['v']; idx=np.arange(len(P));
for name,Q in [('no quadratic',np.empty((len(P),0))),('squares',P**2),('cyclic products',P*np.roll(P,-1,axis=1)),('opposite products',P[:,:2]*P[:,2:]),('shared squares',np.sum(P**2,axis=1)[:,None]),('shared cyclic',np.sum(P*np.roll(P,-1,axis=1),axis=1)[:,None])]:
 def design(k):return np.column_stack((np.ones(len(P)),P,np.sin(P*k),Q))
 errs=[]; coeff=None
 for a1 in [np.array([],int),*np.array_split(idx,6)]:
  a0=np.setdiff1d(idx,a1); fun=lambda k: (lambda X:(X@np.linalg.lstsq(X,V[a0],rcond=None)[0]-V[a0]).ravel())(design(k)[a0]); k=least_squares(fun,np.full(4,2.8),bounds=(.35,6)).x; X=design(k); coeff=np.linalg.lstsq(X[a0],V[a0],rcond=None)[0]; at=a1 if len(a1) else a0; errs.append(np.mean(abs(X[at]@coeff-V[at])));
 print('panel B',name,'fit, oof, worst',np.round([errs[0],np.mean(errs[1:]),max(errs[1:])],6))
