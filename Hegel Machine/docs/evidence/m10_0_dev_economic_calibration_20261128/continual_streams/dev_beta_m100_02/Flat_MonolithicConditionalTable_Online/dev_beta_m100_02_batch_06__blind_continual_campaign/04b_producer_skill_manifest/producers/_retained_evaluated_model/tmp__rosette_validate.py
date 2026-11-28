import importlib.util,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; clean=(abs(y)<5).all(axis=1)
from pathlib import Path
prior=next(Path('/tmp/.hegel_candidate_archive/v01_6868c980').rglob('batch_solver.py'))
for p in (prior,'/holdout_task/batch_solver.py'):
 spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 fit=m._sweep_fit(s); err=[]; idx=np.arange(len(x)); rng=np.random.default_rng(41)
 for inds in np.array_split(rng.permutation(idx[clean]),5):
  use=~np.isin(idx,inds); err.append(np.max(abs(m._raw(x[inds],m._sweep_fit(s,use))-y[inds]),axis=0))
 z=m._calibrate(s).reshape(-1,4); out=m.solve_batch(s); k=out['kkt_certificate']; w=out['weights']; print(p,'fit',fit,'held-out maxima',np.max(err,axis=0),'inverse raw error',np.max(abs(m._raw(z,fit).ravel()-s['y_obs_raw'])),'w',w,'gap',k['primal_objective']-k['dual_lower_bound'],'terminal norm',np.linalg.norm(out['terminal_field']))
