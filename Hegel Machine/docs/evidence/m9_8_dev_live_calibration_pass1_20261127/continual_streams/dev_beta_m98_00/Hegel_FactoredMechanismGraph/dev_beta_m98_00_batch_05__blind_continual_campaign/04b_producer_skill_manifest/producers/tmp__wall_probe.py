import importlib.util,pickle,numpy as np
p='/tmp/hegel_continual_archive/dev_beta_m98_00_batch_02_solver.py'
a=importlib.util.spec_from_file_location('prev',p); m=importlib.util.module_from_spec(a); a.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; modes=s['source_modes']; z=s['Z_nuisance']; y=m._calibration(s['sensor_calibration_sweeps'])(s['y_obs_raw']); P=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); Q=np.eye(len(z))-z@P
from scipy.optimize import least_squares
np.set_printoptions(precision=6,suppress=True)
print('calibrated RMS',np.linalg.norm(y)/np.sqrt(len(y)), 'nuisance-free RMS',np.std(y-z@np.linalg.lstsq(z,y,rcond=None)[0])); print('sensors by axis',np.bincount(s['sensor_ij'][:,0]),np.bincount(s['sensor_ij'][:,1]))
cap=s['u_box']; total=s['total_budget']; ridge=s['source_ridge']; n=len(modes)
def fit(ids, rows):
 zt=z[rows]; q=np.eye(rows.sum())-zt@np.linalg.solve(zt.T@zt+s['nuisance_ridge']*np.eye(zt.shape[1]),zt.T); yt=q@y[rows]; bounds=(np.full(len(ids),0.02),np.full(len(ids),8.0))
 def calc(params, report=False):
  wall=np.ones(8); wall[list(ids)]=params; A=m._response(s,wall,modes)[0]; X=q@A[rows]; w=m._weights(X,yt,ridge,cap,total); r=X@w-yt
  if not report:return np.r_[r,np.sqrt(ridge)*w]
  gamma=np.linalg.solve(zt.T@zt+s['nuisance_ridge']*np.eye(zt.shape[1]),zt.T@(y[rows]-A[rows]@w)); err=y-A@w-z@gamma
  return wall,w,np.sqrt(np.mean(r*r)),np.sqrt(np.mean(err[~rows]**2)) if not rows.all() else np.nan
 args=least_squares(calc,np.ones(len(ids)),bounds=bounds,max_nfev=45).x if ids else []
 return calc(args,True)
allrows=np.ones(len(y),dtype=bool)
for ids in [(),(0,),(1,),(4,),(5,),(0,1),(4,5),(0,1,4,5),(2,3,6,7),tuple(range(8))]:
 wall,w,rmse,_=fit(ids,allrows); cvs=[fit(ids,np.arange(len(y))//4%4!=j)[-1] for j in range(4)]; print(ids,'wall',wall,'w',w,'rmse',rmse,'rosette-heldout',np.round(cvs,7))
