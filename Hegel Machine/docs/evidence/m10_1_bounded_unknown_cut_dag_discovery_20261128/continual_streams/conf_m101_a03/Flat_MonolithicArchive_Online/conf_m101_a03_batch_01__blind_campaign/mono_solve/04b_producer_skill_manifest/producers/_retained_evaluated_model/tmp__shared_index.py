import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import _affine as af,_fit as fit,_coordinates,_polynomial as poly
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; w=_coordinates(s,t['x']); a=af(w); p=poly(w,2)[:,5:]; pairs=np.c_[w[:,0]*w[:,2],w[:,1]*w[:,3]]
rng=np.random.default_rng(127); splits=[(np.arange(k),np.arange(k,len(y))) for k in (52,56)]; splits += [(q[:52],q[52:]) for q in [rng.permutation(len(y)) for _ in range(12)]]; splits += [(q[:52],q[52:]) for q in [np.argsort(w[:,k]*d) for k in range(4) for d in (-1,1)]]
print('trajectory coefficients',fit(a,t['traj']).round(3)); print('test ranges',np.max(abs(_coordinates(s,s['test']['x'])),0))
for kind in ('abs','square','sqrt','tanh','atan','asinh'):
 for mode in ('inside','outside','both','sparse_inside','sparse_outside','sparse_both'):
  errs=[]; details=None
  for i,j in splits:
   initial=fit(a[i],y[i])
   def features(c):
    v=a@c; av=abs(v); d=1+av if kind=='abs' else 1+v*v if kind=='square' else np.sqrt(1+v*v) if kind=='sqrt' else np.divide(v,np.tanh(v),out=np.ones_like(v),where=av>1e-12) if kind=='tanh' else np.divide(v,np.arctan(v),out=np.ones_like(v),where=av>1e-12) if kind=='atan' else np.divide(v,np.arcsinh(v),out=np.ones_like(v),where=av>1e-12)
    q=pairs if mode.startswith('sparse') else p; z=np.c_[a,1/d,v/d]; u=q/d[:,None]; loc=mode.split('_')[-1]; return np.c_[z, u if loc=='inside' else q if loc=='outside' else np.c_[q,u]]
   def run(c):
    F=features(c); return F@fit(F[i],y[i])
   ini=initial*(.2 if kind in ('abs','square') else .8)
   res=least_squares(lambda c: (run(c)-y)[i],ini,max_nfev=100); pred=run(res.x); errs.append(np.mean(abs(pred[j]-y[j])))
   if details is None: details=(np.std((pred-y)[i]),res.x.round(3))
  print(kind,mode,np.round([*errs[:2],np.mean(errs[2:14]),max(errs[2:14]),max(errs[14:])],5),details,flush=True)
