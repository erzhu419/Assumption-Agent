exec(open('/tmp/readout_probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import minimize_scalar,least_squares
np.set_printoptions(precision=4,suppress=True)
A=poly(z,1); B=s['panel_b']; k,g=m._fit_b(B['u'],B['v']); H=m._b_design(u,k)[:,5:]; Z=H@g[5:]; T=Y[:,0]; J=np.arange(4); perms=list(itertools.permutations(J));
# Every dominant mixed term in the earlier label search used a different channel on both sides.
results=[]
for name,q in [('z',z),('Z',Z),('H',H)]:
 for basename,base in [('z',A),('z,u',np.c_[A,u]),('u',poly(u,1)),('z2',poly(z,2)),('z,u2',np.c_[A,poly(u,2)[:,1:]]),('z,Z,u',np.c_[A,Z,u])]:
  for form in range(6):
   def model(a,b):
    best=None
    for p in perms:
     w=u*q[:,p];w=[w,w**2,np.c_[w,w**2],np.c_[w,np.abs(w)], np.c_[w,np.tanh(2*w)],np.c_[w,np.sign(w)*abs(w)**1.5]][form]; d=np.c_[base,w]; Q=np.linalg.svd(d[a],full_matrices=False)[0]; r=(T[a]-Q@(Q.T@T[a]))/np.maximum(1-(Q*Q).sum(1),1e-10); loss=np.mean(r*r)
     if best is None or loss<best[0]:best=(loss,p,d)
    loss,p,d=best;c=np.linalg.lstsq(d[a],T[a],rcond=None)[0];return np.mean(abs(d[b]@c-T[b])),loss,p,c
   vals=[model(a,b) for a,b in fs];results.append((max(v[0] for v in vals),vals[0][0],name,basename,form,vals[0][2],vals[0][3],vals[0][1]))
for row in sorted(results,key=lambda r:r[0])[:12]:print('best',row)
# Fit a common nonlinear scale to the four products while retaining their sparse pairing.
for name,q in [('z',z),('Z',Z),('H',H)]:
 w=u*q[:,[3,0,1,2]]
 for bname,d0 in [('z',A),('z,u',np.c_[A,u]),('u',poly(u,1)),('z2',poly(z,2))]:
  for fname,fun in [('tanh',np.tanh),('sin',np.sin),('asinh',np.arcsinh),('rational',lambda x:x/(1+x*x)),('exp',lambda x:np.expm1(np.clip(x,-10,10)))]:
   errs=[]; ks=[]
   for a,b in fs:
    def pred(k,ii):
     d=np.c_[d0,fun(k*w)];c=np.linalg.lstsq(d[a],T[a],rcond=None)[0];return d[ii]@c
    opt=minimize_scalar(lambda k:np.mean((pred(k,a)-T[a])**2),bounds=(.005,4),method='bounded');ks.append(opt.x);errs.append(np.mean(abs(pred(opt.x,b)-T[b])))
   if max(errs)<.075 or errs[0]<.042:print('sparse nonlinear',name,bname,fname,np.round(errs,4),np.round(ks,3))
# Inspect stable full-design bilinear coefficients for both readouts.
for q in [z,Z,H]:
 d=np.c_[A,u,(u[:,:,None]*q[:,None,:]).reshape(len(u),16)];c=np.linalg.lstsq(d,Y,rcond=None)[0];print('bilinear y, trajectory endpoint',c[9:,0].reshape(4,4),c[9:,-1].reshape(4,4),np.std(Y-d@c,axis=0))
