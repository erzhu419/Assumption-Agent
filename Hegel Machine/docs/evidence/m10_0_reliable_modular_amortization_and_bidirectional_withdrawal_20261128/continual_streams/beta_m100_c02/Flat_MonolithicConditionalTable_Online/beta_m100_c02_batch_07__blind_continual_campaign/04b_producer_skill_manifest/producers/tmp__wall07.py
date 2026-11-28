exec(open('/tmp/probe07.py').read().split("for kind in ('add','poly'):")[0])
from scipy import sparse
from scipy.sparse.linalg import splu
B=fit(lambda x:additive(x,1),np.ones(len(X),bool)); print('affine',B)
o=np.empty(4); a=np.empty(4)
for k in range(4):
  o[k],a[k]=np.linalg.lstsq(np.c_[np.ones(len(X)+(j==k).sum()),np.r_[X[:,k],q['ref_signal'][j==k]]],np.r_[Y[:,k],vs[j==k]],rcond=None)[0]
yp=(YP-o)/a
nx,ny=s['nx'],s['ny']; dx=s['Lx']/(nx-1);dy=s['Ly']/(ny-1); I=sparse.eye(nx*ny,format='csc'); modes=np.asarray(s['source_modes']).reshape(-1,nx*ny,order='F'); ij=s['sensor_ij']; ids=ij[:,0]+nx*ij[:,1]; rhs=modes.T@W.T
axes=[]; pars=[]
for axis,h in [('x',dx),('y',dy)]:
 kap=s['kappa_'+axis]; vel=s['vel_'+axis]; n=len(kap); axes.append(base.axis_operator(kap,vel,h,[0,0])); p=per.axis_operator(kap,vel,h,periodic=True).toarray(); pars.append(np.array([[ -p[0,1],-p[0,-1],0],[-p[-1,-2],-p[-1,0],0]])); print(axis,n,h,'Robin',s['robin_'+axis], 'k,h-scaled',kap[[0,-1]]/h**2,'velocity',vel[[0,-1]], 'periodic',pars[-1]);
sealed=np.array([[[s['kappa_x'][0]/dx**2,0,0],[s['kappa_x'][-1]/dx**2,0,0]],[[s['kappa_y'][0]/dy**2,0,0],[s['kappa_y'][-1]/dy**2,0,0]]]); robin=sealed.copy();robin[:,:,2]=np.array([s['robin_x']/dx,s['robin_y']/dy])[:,None]; periodic=np.array(pars)
def pred(p):
 out=[]
 for A,p0 in zip(axes,np.reshape(p,(2,2,3))):
  n=A.shape[0]; A=A.tolil();
  for i,near,far,(inward,wrap,loss) in zip((0,n-1),(1,n-2),(n-1,0),p0):
   A[i,i]=inward+wrap+loss; A[i,near]=-inward; A[i,far]=-wrap
  out.append(A.tocsc())
 L=sparse.kron(sparse.eye(ny),out[0],format='csc')+sparse.kron(out[1],sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
 return np.concatenate([splu(L+mu*I).solve(rhs)[ids].T for mu in s['mu_shifts']],axis=1).reshape(yp.shape)
def score(e):return np.sqrt(np.mean(e**2,axis=(1,2)))
for i,(u,v) in enumerate(itertools.product((sealed,robin,periodic),repeat=2)):
 p=np.stack([u[0],v[1]]); e=pred(p)-yp;print('combo',i,score(e),np.max(abs(e)))
for k in [None,0,1,2]:
 sel=np.array([i!=k for i in range(len(yp))]); fitp=least_squares(lambda p:(pred(p)-yp)[sel].ravel(),((sealed+periodic)/2).ravel(),ftol=1e-11,gtol=1e-12,x_scale='jac',max_nfev=100).x
 print('fit drop',k,'RMSE',score(pred(fitp)-yp),'params',fitp.reshape(2,2,3))
