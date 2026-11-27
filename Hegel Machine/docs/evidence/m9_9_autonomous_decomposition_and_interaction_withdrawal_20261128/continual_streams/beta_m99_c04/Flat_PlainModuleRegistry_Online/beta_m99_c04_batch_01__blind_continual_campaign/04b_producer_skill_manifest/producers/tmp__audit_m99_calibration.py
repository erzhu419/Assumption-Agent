import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
batch = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))
print('public studies:', len(batch))
for item in batch:
 s = item['study']; sweep = s['sensor_calibration_sweeps']; x = sweep['ref_rosette']; y = sweep['raw_rosette']; S, _ = m._design(s)
 def basis(x, kind):
  linear = np.c_[np.ones(len(x)), x]
  if kind == 'affine': return linear
  p = np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
  if kind == 'pairwise-quadratic': return np.c_[linear,p]
  return np.c_[linear, x / (1+x*x), np.column_stack([(x[:,i]-x[:,j])/(1+(x[:,i]-x[:,j])**2) for i in range(4) for j in range(i+1,4)])]
 rng = np.random.default_rng(2026)
 for kind in ('affine','pairwise-quadratic','rational-pairwise'):
  a=basis(x,kind); e=np.empty_like(y)
  for test in np.array_split(rng.permutation(len(x)) if kind=='affine' else np.random.default_rng(2026).permutation(len(x)),5):
   train=np.setdiff1d(np.arange(len(x)),test); e[test]= a[test]@np.linalg.lstsq(a[train],y[train],rcond=None)[0]-y[test]
  print(kind, 'pooled RMSE', np.sqrt(np.mean(e*e)), 'worst channel RMSE', np.sqrt(np.mean(e*e,axis=0)).max())
 a=basis(x,'affine'); C=np.linalg.lstsq(a,y,rcond=None)[0]
 def forward(v): return (v.reshape(-1,4)@C[1:]+C[0]).ravel()
 ij=s['sensor_ij']; wall=(ij[:,0]==0)|(ij[:,0]==s['nx']-1)|(ij[:,1]==0)|(ij[:,1]==s['ny']-1)
 Z=s['Z_nuisance']; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
 for j,p in enumerate(item.get('campaign_diagnostics') or []):
  pred=S@p['w_pulse']; raw=(forward(pred)-p['y_pulse_raw']).reshape(len(s['mu_shifts']),-1)
  cal=m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw']))-pred
  r=cal.reshape(raw.shape); proj=(cal-Z@(P@cal)).reshape(r.shape)
  rmse=lambda v: np.sqrt(np.mean(v*v,axis=1))
  print('pulse',j,'worst raw channel',np.sqrt(np.mean(raw.reshape(-1,4)**2,axis=0)).max(),'worst wall/interior shift',rmse(r[:,wall]).max(),rmse(r[:,~wall]).max(),'worst projected shift',rmse(proj).max())
 out=m.solve_batch(s); w=np.asarray(out['weights']); obs=m._calibrate(s); err=obs-S@w; zpart=Z@(P@err)
 print('nuisance rank',np.linalg.matrix_rank(Z),'observation residual RMSE before / after required projection',np.sqrt(np.mean(err**2)),np.sqrt(np.mean((err-zpart)**2)))
