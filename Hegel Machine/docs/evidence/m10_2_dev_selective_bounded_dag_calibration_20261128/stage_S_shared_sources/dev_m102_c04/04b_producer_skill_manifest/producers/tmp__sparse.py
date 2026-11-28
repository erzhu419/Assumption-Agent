exec(open('/tmp/structure.py').read().split('np.set_printoptions')[0]); np.set_printoptions(precision=5,suppress=True)
d=p*r; ij=[(i,j) for i in range(4) for j in range(4) if i!=j]; off=np.column_stack([p[:,i]*r[:,j] for i,j in ij]); A=np.c_[p,r,d.sum(1),d[:,:3]-d[:,[3]]]; V=-np.expm1(-t[:,None]*[1.805,.427]); print('off diagonal order',ij); print('coefficients',m._ls(m._affine(np.c_[A,off]),np.c_[y,m._ls(V,z.T).T]).T); print('scalar constrained',m._ls(m._affine(A[:,:9]),y));
split=(np.arange(52),np.arange(52,72)); folds=[split]; rng=np.random.default_rng(671)
for _ in range(30):
 i=rng.permutation(72); folds.append((i[:52],i[52:]))
for a in [*x.T,np.linalg.norm(x,axis=1)]:
 for b in [-1,1]:
  i=np.argsort(b*a); folds.append((i[:52],i[52:]))
bs=[m._temporal_basis(t,z[i]) for i,j in folds];
for name,F in [('y pr summed diag',A[:,:9]),('y pr diag',A)]:
 E=[np.mean(abs(m._ridge(F[i],y[i],F[j],1e-6)-y[j])) for i,j in folds]; print(name,E[0],max(E[1:31]),max(E[31:]))
for name,F,penalty in [('outer',np.c_[A,off],np.ones(24)),('common diag plus off',np.c_[A[:,:9],off],np.ones(21)),('common diag plus two off',np.c_[A[:,:9],off[:,[0,1]]],np.ones(11)),('common diag plus one off',np.c_[A[:,:9],off[:,0]],np.ones(10)),('diag plus two off',np.c_[A,off[:,[0,1]]],np.ones(14)),('weighted outer',np.c_[A,off],np.r_[np.ones(9),np.full(3,80),np.ones(2),np.full(10,80)])]:
 for lam in [1e-7,1e-5,1e-4,.001]:
  E=[]
  for (i,j),bas in zip(folds,bs):
   f=F/np.sqrt(penalty); err=abs(m._ridge(f[i],z[i]@bas,f[j],lam)@bas.T-z[j]); E.append([err.mean(),err[:,-1].mean()]);
  E=np.array(E); print(name,lam,E[0],E[1:31].max(0),E[31:].max(0))
