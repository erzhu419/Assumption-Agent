from probe import *
b=s['panel_b']; W=ls(np.c_[np.ones(len(b['u'])),b['u']],b['v']); print('panel B singular values',np.linalg.svd(W[1:],compute_uv=False))
fwd=lambda x:x@W[1:]+W[0]; inv=lambda x:np.linalg.solve(W[1:].T,(x-W[0]).T).T
v1basis=lambda a:np.c_[np.ones(len(a)),a[:,0]*a[:,1],a[:,0]*a[:,3],a[:,1]*a[:,2],a[:,2]*a[:,3]]
v2basis=lambda a:np.c_[np.ones(len(a)),np.cos(2.4*a)]
p1=ls(v1basis(u),v1);p2=ls(v2basis(u),v2)
print('panel LOO',loo(v1basis(u),v1),loo(v2basis(u),v2)); print('readout coefficients',p1,p2)
coords={'raw':x,'Ainv':z,'B':fwd(x),'Binv':inv(x),'Ainv_B':fwd(z),'Ainv_Binv':inv(z),'B_Ainv':np.linalg.solve(w[1:].T,(fwd(x)-w[0]).T).T,'Binv_Ainv':np.linalg.solve(w[1:].T,(inv(x)-w[0]).T).T}
fam=[{n:fn(a)@coef for n,a in coords.items()} for fn,coef in [(v1basis,p1),(v2basis,p2)]]
res=[]
for n,g in fam[0].items():
 for m,h in fam[1].items():
  for kind,F in [('readouts',np.c_[g,h]),('plus coordinates',np.c_[g,h,z])]:
   v=ev(F);res.append((v.max(),n,m,kind,v[0],v.max(0)))
for row in sorted(res,key=lambda v:v[0])[:22]:print(row)
