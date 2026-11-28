exec(open('/tmp/probe_readout.py').read().split("print('warp'")[0])
from scipy.optimize import minimize_scalar
from itertools import product
idx=np.arange(n)
folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(917).permutation(n),5))+[np.argsort(np.linalg.norm(x,axis=1))[-14:]]
u=x[:,1]-x[:,3]; P=np.c_[z[:,0]*z[:,2],z[:,1]*z[:,3]]
warpf,warpa,W=b['_fit_panel'](s['panel_b']['u'],s['panel_b']['v']); print('operator',warpf,warpa,np.round(W,3))
for label,B in [('z',np.c_[np.ones(n),z]),('xz',np.c_[np.ones(n),x,z]),('trajectory',np.c_[np.ones(n),z,T[:,[-1]]])]:
 for i,j in product(range(2),repeat=2):
  A=np.c_[B,P[:,np.array([i,j],bool)]]
  for name,fun in [('sin',np.sin),('tanh',np.tanh)]:
   E=[]; pars=[]
   for v in folds:
    t=np.setdiff1d(idx,v)
    def calc(f,ret=False):
     F=np.c_[A,fun(f*u)]; w=np.linalg.lstsq(F[t],y[t],rcond=None)[0]; r=F@w-y
     return (np.mean(abs(r[v])), f, w,np.std(r[t])) if ret else r[t]@r[t]
    grid=np.linspace(.25,6,47); j0=np.argmin([calc(f) for f in grid]); f=minimize_scalar(calc,bounds=(grid[max(j0-1,0)],grid[min(j0+1,len(grid)-1)]),method='bounded').x
    e,f,w,tr=calc(f,True); E.append(e); pars.append(f)
   if i==j or max(E)<.052: print(label,i,j,name,np.round(E,4), 'freq',np.round(pars,2),'coef',np.round(w,3),'train',round(tr,4))
# Inspect centered trajectory shapes and whether a quadratic operator correction could matter.
print('traj singular',np.round(np.linalg.svd(T-T.mean(0),compute_uv=False),4))
print('y/terminal correlation',np.corrcoef(y,T[:,-1])[0,1]); print('operator residual scalar',np.std(y-np.c_[np.ones(n),z]@np.linalg.lstsq(np.c_[np.ones(n),z],y,rcond=None)[0]))
