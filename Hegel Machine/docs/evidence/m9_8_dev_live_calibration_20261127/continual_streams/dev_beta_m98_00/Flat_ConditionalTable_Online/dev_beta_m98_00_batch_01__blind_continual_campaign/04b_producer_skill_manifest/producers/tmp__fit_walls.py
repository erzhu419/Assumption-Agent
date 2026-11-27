import importlib.util,pickle,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_beta_m98_00_s00_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=b['study']; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; h=[s['Lx']/(nx-1),s['Ly']/(ny-1)]; ks=[s['kappa_x'],s['kappa_y']]; vs=[s['vel_x'],s['vel_y']]; rb=[s['robin_x'],s['robin_y']]; ps=b['campaign_diagnostics']; w=np.array([p['w_pulse'] for p in ps]).T; y=np.array([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in ps]).T; f=np.column_stack([u.ravel(order='F') for u in s['source_modes']])@w; z=s['Z_nuisance']; q=np.linalg.qr(z)[0]; eye=sparse.eye(nx*ny,format='csc'); reaction=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
print('shifts',s['mu_shifts'],'pulse budgets',w.sum(0),'pulse RMS',np.mean(y*y,axis=0)**.5)
for j in range(2): print('axis',j,'h',h[j],'kap wall',ks[j][[0,1,-2,-1]],'vel wall',vs[j][[0,-1]],'robin',rb[j])
def pred(p):
 a=[]
 for j in range(2):
  mat=m._axis(ks[j],vs[j],h[j],rb[j]).tolil(); n=mat.shape[0]
  for side,(i,near,far) in enumerate(((0,1,n-1),(n-1,n-2,0))):
   d,t,l=p.reshape(4,3)[2*j+side]; mat[i,i]=d+t+l; mat[i,near]=-d; mat[i,far]=-l
  a.append(mat.tocsc())
 op=sparse.kron(sparse.eye(ny),a[0],format='csc')+sparse.kron(a[1],sparse.eye(nx),format='csc')+reaction
 return np.vstack([splu(op+mu*eye).solve(f)[rows] for mu in s['mu_shifts']])
def res(p,strip=False):
 r=pred(p)-y
 return r-q@(q.T@r) if strip else r
def stats(r):
 return np.sqrt(np.mean(r*r,axis=0)),np.max(abs(r),axis=0)
base=np.array([[ks[j][i]/h[j]**2,rb[j]/h[j],0] for j in range(2) for i in (0,-1)]); face=np.array([[.5*(ks[j][i]+ks[j][near])/h[j]**2,0,0] for j in range(2) for i,near in ((0,1),(-1,-2))]);
for name,p in [('Robin',base),('reflect',base*[1,0,0]),('reflectFace',face),('doubleFace',2*face),('periodic',face+np.array([[0,0,.5*(ks[j][0]+ks[j][-1])/h[j]**2] for j in range(2) for _ in (0,1)]))]: print(name,stats(res(p)))
for kind in ('local','wrap'):
 for train in ([0],[1],[0,1]):
  for strip in (False,True):
   def expand(p):
    return np.c_[p.reshape(4,2),np.zeros(4)] if kind=='local' else p.reshape(4,3)
   init=base[:,:2].ravel() if kind=='local' else base.ravel(); lo=np.tile([.01,-2] if kind=='local' else [.01,-2,-12],4); hi=np.tile([150,80] if kind=='local' else [150,80,150],4)
   fit=least_squares(lambda p: res(expand(p),strip)[:,train].ravel(),init,bounds=(lo,hi),xtol=1e-12,gtol=1e-13,ftol=1e-12,max_nfev=90); print(kind,train,strip, 'coef',np.round(expand(fit.x),5).tolist(),'rmse,max',stats(res(expand(fit.x))), 'projected',stats(res(expand(fit.x),True))[0]);
