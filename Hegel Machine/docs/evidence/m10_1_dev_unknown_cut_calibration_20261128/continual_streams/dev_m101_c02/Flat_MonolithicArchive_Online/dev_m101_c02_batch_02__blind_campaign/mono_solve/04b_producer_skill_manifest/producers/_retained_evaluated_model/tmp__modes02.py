import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=m._coordinates(s,s['train']['x']); T=s['train']['traj']; t=s['eval_times']; y=s['train']['y']
B=np.stack([np.log1p(1.4*t),-np.expm1(-.7*t)]); print('two-mode loadings',np.linalg.lstsq(B.T,np.linalg.lstsq(z,T,rcond=None)[0].T,rcond=None)[0].T.round(5));
for name in ('full','separated','separated_fixed'):
 es=[]; ends=[]; rates=[]
 for vi in [np.arange(56,72)]+[np.random.default_rng(seed).permutation(72)[:16] for seed in range(20,32)]+[np.argsort(np.linalg.norm(z,axis=1))[-16:]]:
  ti=np.setdiff1d(np.arange(72),vi)
  def d(i,p):
   L=np.log1p(p[0]*t); E=-np.expm1(-p[1]*t); a=z[i,:,None];
   return np.concatenate((a*L,a*E),axis=1).transpose(0,2,1).reshape(-1,8) if name=='full' else np.concatenate((a[:,:1]*L,a[:,1:]*E),axis=1).transpose(0,2,1).reshape(-1,4)
  def w(p):return m._fit(d(ti,p),T[ti].ravel())
  p=least_squares(lambda p:d(ti,p)@w(p)-T[ti].ravel(),[1.4,.7],bounds=(.001,20)).x if name!='separated_fixed' else [1.4,.7]; rates.append(p); err=abs((d(vi,p)@w(p)).reshape(-1,len(t))-T[vi]); es.append(err.mean()); ends.append(err[:,-1].mean())
 print(name,'traj mae',np.round(es,6),'endpoint mae',np.round(ends,6),'rate range',np.min(rates,axis=0),np.max(rates,axis=0))
