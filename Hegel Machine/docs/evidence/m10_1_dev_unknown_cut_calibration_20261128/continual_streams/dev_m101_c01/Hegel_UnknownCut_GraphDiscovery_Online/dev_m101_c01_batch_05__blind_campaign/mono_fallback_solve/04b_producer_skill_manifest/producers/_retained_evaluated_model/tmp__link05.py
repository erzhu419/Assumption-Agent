exec(open('/tmp/probe05.py').read().split('p=temp(z[:56])')[0])
from scipy.optimize import minimize_scalar

def quad(x):
 return np.column_stack([np.ones(len(x)),x]+[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)])

def linkprobe(F, obs, ids, js):
 out={}
 beta=np.linalg.lstsq(F[ids],obs[ids],rcond=None)[0]
 out['quadratic']=F@beta
 for name,fn in [('tanh',np.tanh),('sin',np.sin),('asinh',np.arcsinh),('atan',np.arctan),('softsign',lambda q:q/np.sqrt(1+q*q))]:
  mid=obs[ids].mean(); sd=obs[ids].std(); Y=(obs-mid)/sd; G=F[ids]; o=Y[ids]; best=None
  for scale in (2,4,10):
   guess=np.r_[0.,scale,np.linalg.lstsq(G,o/scale,rcond=None)[0]]
   def residual(p): return p[0]+p[1]*fn(G@p[2:])-o
   fit=least_squares(residual,guess,max_nfev=120,ftol=1e-8)
   risk=np.mean(fit.fun**2)
   if best is None or risk<best[0]: best=risk,fit.x
  p=best[1];out[name]=(p[0]+p[1]*fn(F@p[2:]))*sd+mid
 return {k:(np.mean(abs(v[ids]-obs[ids])),np.mean(abs(v[js]-obs[js]))) for k,v in out.items()}

print('terminal_y difference',np.max(abs(y-z[:,-1])), 'times', t)
for name in ('v1','v2'):
 panel=s['panel_c']; F=quad(panel['u']); stats=[]
 for seed in range(3):
  r=np.random.default_rng(seed).permutation(len(F)); stats.append([linkprobe(F,o,r[:19],r[19:]) for o in panel[name].T])
 print('panel C',name,{k:np.mean([a[k] for fold in stats for a in fold],axis=0) for k in stats[0][0]})
q=a(x); F=quad(q); rng=np.random.default_rng(211); folds=[(np.arange(56),np.arange(56,72))]+[(r[:56],r[56:]) for r in (rng.permutation(72) for _ in range(4))]
vals=[]
for i,j in folds:
 p=temp(z[i]); v=amp(z,p); vals.append([linkprobe(F,o,i,j) for o in np.c_[y,v].T])
for h in range(3):
 print('response',h,{k:(np.mean([a[h][k][0] for a in vals]),vals[0][h][k][1],np.mean([a[h][k][1] for a in vals]),max(a[h][k][1] for a in vals)) for k in vals[0][0]})
