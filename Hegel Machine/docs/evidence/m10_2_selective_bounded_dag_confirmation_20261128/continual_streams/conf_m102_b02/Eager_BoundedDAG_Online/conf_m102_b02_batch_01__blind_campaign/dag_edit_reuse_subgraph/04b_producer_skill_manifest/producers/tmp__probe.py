import numpy as np,pickle,runpy
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b02_batch_00_solver.py'); np.set_printoptions(precision=4,suppress=True)
print('telemetry',s['telemetry_vec'],'summary',s['train']['summary']); print('all study keys',s.keys()); print('trajectory spectrum',np.linalg.svd(s['train']['traj'],compute_uv=False))
a=m['_affine'](s['panel_a'],True); x=np.vstack([s['train']['x'],s['test']['x']]); x=np.linalg.solve(a[1:].T,(x-a[0]).T).T
print('latent min and max',x.min(0),x.max(0))
n=72; tar=np.c_[s['train']['y'],s['train']['traj']]; folds=[np.arange(52,72),*np.array_split(np.random.default_rng(42).permutation(n),4)]
def score(z):
 G=m['_kernel'](z[:n],z[:n]); res=[]
 for j in folds:
  i=np.setdiff1d(np.arange(n),j); g=G[np.ix_(i,i)]; g.flat[::len(i)+1]+=1e-6*np.diag(g).mean(); modes=np.linalg.svd(tar[i,1:],full_matrices=False)[2][:2]; p=G[np.ix_(j,i)]@np.linalg.solve(g,np.c_[tar[i,0],tar[i,1:]@modes.T]); r=abs(np.c_[p[:,0],p[:,1:]@modes]-tar[j]); res.append([r[:,0].mean(),r[:,1:].mean()]);
 return np.r_[res[0],np.max(res,axis=0)]
def poly(x,k):
 return np.column_stack([np.ones(len(x)),*[np.prod(x[:,t],axis=1) for d in range(1,k+1) for t in combinations_with_replacement(range(4),d)]])
print('raw',score(x))
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[p]['u']; v=s[p][k]; print('\n',p,k,'standardization',v.min(0),v.max(0));
 for deg in (1,2,3):
  D=poly(u,deg); C=np.linalg.lstsq(D,v,rcond=None)[0]; H=D@np.linalg.pinv(D); r=D@C-v; print('poly',deg,'rms',np.mean(r*r)**.5,'loo',np.mean((r/np.maximum(1e-10,1-H.diagonal())[:,None])**2)**.5,'target',score(poly(x,deg)@C));
  if deg==2: print('quadratic coefficients',C[5:]); print('quad residual norms',np.linalg.norm(r,axis=1))
 D=poly(u,1); X=poly(x,1); base=np.linalg.lstsq(D,v,rcond=None)[0]
 for name,f in [('tanh',np.tanh),('asinh',np.arcsinh),('sin',np.sin),('sat',lambda t:t/np.sqrt(1+t*t)),('rational',lambda t:t/(1+t*t)),('cube',lambda t:t+t**3)]:
  preds=[]; errs=[]; params=[]
  for col in range(4):
   y=v[:,col]; fits=[least_squares(lambda w:w[0]+np.exp(w[1])*f(D@w[2:])-y,np.r_[0.,np.log(sc),base[:,col]/sc],max_nfev=250) for sc in (0.5,1,2)]; w=min(fits,key=lambda q:q.cost).x; preds.append(w[0]+np.exp(w[1])*f(X@w[2:])); errs.append(np.std(w[0]+np.exp(w[1])*f(D@w[2:])-y)); params.append([w[0],np.exp(w[1])]);
  print(name,np.round(errs,5),score(np.column_stack(preds)),'scale',np.round(params,3))
