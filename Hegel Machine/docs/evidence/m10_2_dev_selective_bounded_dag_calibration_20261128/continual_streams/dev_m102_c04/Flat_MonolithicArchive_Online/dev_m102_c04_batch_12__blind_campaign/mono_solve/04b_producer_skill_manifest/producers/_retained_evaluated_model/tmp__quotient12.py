exec(open('/tmp/audit12.py').read().split('for label,z0 in')[0])
from scipy.optimize import least_squares
cross=p[:,0]*r[:,1]
# Keep the scalar readout fixed; only the state readout has evidence for this term.
q=z[:,2]-.5*z[:,3]

def evaluate_state(tr,te,scheme):
 order=np.r_[tr,te]; f=F[order]; g=S[order]; h=q[order]; c=cross[order]; k=len(tr); target=a[tr].ravel()
 def calc(par):
  b=m._basis(times,par[:2]); ff=f
  if scheme=='fixed': ff=m._design(f[:,1:],c/(1+.25*h*h))
  if scheme=='profiled':
   q1=z[order,2]+par[3]*z[order,3]
   ff=m._design(f[:,1:],c/(1+par[2]*q1*q1))
  d=np.column_stack(((ff[:k,None,:]*b[None,:,0,None]).reshape(len(target),-1),(g[:k,None,:]*b[None,:,1,None]).reshape(len(target),-1)))
  w=m._fit(d,target)
  return d@w-target,ff,w,b
 start=[1.8,.43]; lo=[.8,.03]; hi=[5,1.]
 if scheme=='profiled': start += [.25,-.5]; lo += [.005,-3]; hi += [4,3]
 opt=least_squares(lambda par:calc(par)[0],start,bounds=(lo,hi)); resid,ff,w,b=calc(opt.x); j=ff.shape[1]
 pred=np.column_stack((ff[k:]@w[:j],g[k:]@w[j:]))@b.T
 err=abs(pred-a[te]); return np.array([err.mean(),err.mean(0).max(),err[:,-1].mean()]),opt.x,w[j-1]
idx=np.arange(len(x)); rng=np.random.default_rng(1207)
groups={'required':[held],'random':[rng.choice(idx,20,replace=False) for _ in range(30)],'extremes':[np.argsort(col)[sl] for col in np.c_[z,np.linalg.norm(z,axis=1),np.sum(p*r,axis=1),cross,q].T for sl in (slice(None,20),slice(-20,None))]}
print('Metrics: trajectory MAE, worst-horizon MAE, terminal MAE')
for scheme in ('base','fixed','profiled'):
 print(scheme)
 for label,folds in groups.items():
  runs=[evaluate_state(np.setdiff1d(idx,te),te,scheme) for te in folds]; er=np.array([v[0] for v in runs]); coef=np.array([v[2] for v in runs]); params=np.array([v[1] for v in runs]); print(label,'max',er.max(0),'mean',er.mean(0),'last fast coefficient quantiles',np.quantile(coef,[0,.5,1]),'parameter quantiles',np.quantile(params,[0,.5,1],axis=0))
