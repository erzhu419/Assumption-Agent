exec(open('/tmp/compare_walls.py').read().split('for a,b in itertools.product')[0])
for a,b in [('P','P'),('P','R'),('R','P'),('R','R')]:
 l=kron(eye(ny),axis('x',a))+kron(axis('y',b),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')); A=np.concatenate([splu((l+m*eye(nx*ny)).tocsc()).solve(rhs)[x+nx*yidx] for m in mus]); w,g=fit(A,np.ones(len(y),bool)); err=A@w+z@g-y
 B=np.c_[A[:,:-1]-A[:,-1:],z]; target=y-budget*A[:,-1]; co=np.linalg.lstsq(B,target,rcond=None)[0]; e=B@co-target
 print(a+b,'unpenalized residual rms',np.linalg.norm(e)/np.sqrt(len(y)),'weights',np.round(np.r_[co[:len(modes)-1],budget-co[:len(modes)-1].sum()],4))
 for name,idx in [('xwall',(x==0)|(x==nx-1)),('ywall',(yidx==0)|(yidx==ny-1)),('interior',(x>0)&(x<nx-1)&(yidx>0)&(yidx<ny-1))]:
  r=e.reshape(len(mus),-1)[:,idx]; print(name,np.sqrt(np.mean(r*r,axis=1)).round(6),r.mean(axis=1).round(6))
 print('unpenalized per bank',np.sqrt(np.mean(e.reshape(-1,4)**2,axis=0)).round(6))
