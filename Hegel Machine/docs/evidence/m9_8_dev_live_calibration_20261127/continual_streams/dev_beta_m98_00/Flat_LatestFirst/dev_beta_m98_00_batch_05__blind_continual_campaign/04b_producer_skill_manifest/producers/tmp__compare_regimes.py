import glob, importlib.util, pickle
import numpy as np
from scipy.optimize import minimize
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
q=s['sensor_calibration_sweeps']; ref=q['ref_rosette']; raw=q['raw_rosette']; x=np.c_[np.ones(len(ref)),ref]; b=np.linalg.lstsq(x,raw,rcond=None)[0]
e=raw-x@b; hat=np.sum(x*np.linalg.pinv(x).T,axis=1)
print('calibration max leave-one-sweep-out error by channel',np.max(abs(e/(1-hat[:,None])),axis=0)); print('bank sweep max error',np.max(abs(q['single_bank_sweeps']['raw_readout']-b[0,q['single_bank_sweeps']['bank_id']]-np.diag(b[1:])[q['single_bank_sweeps']['bank_id']]*q['single_bank_sweeps']['ref_signal'])))
y=np.linalg.solve(b[1:].T,(s['y_obs_raw'].reshape(-1,4)-b[0]).T).T.ravel(); z=s['Z_nuisance']; n=len(y); cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']; m=len(s['source_modes']); nb=n//4
for p in sorted(glob.glob('/tmp/hegel_continual_archive/*_solver.py')):
 sp=importlib.util.spec_from_file_location('arch',p); a=importlib.util.module_from_spec(sp); sp.loader.exec_module(a)
 S,U=a._design(s); errs=[]; gs=[]
 for i in range(-1,nb):
  keep=(np.arange(n)//4!=i); D=S[keep]; Y=y[keep]; Z=z[keep]; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(z.shape[1]),Z.T)
  Dt=D-Z@(P@D); Yt=Y-Z@(P@Y)
  w,nu=a._box_qp(Dt.T@Dt+ridge*np.eye(m),-Dt.T@Yt,cap,budget); gamma=P@(Y-D@w)
  if i<0:
   r=y-S@w-z@gamma; print(p.split('/')[-1], 'design norm',np.linalg.norm(S),'fit RMS',np.sqrt(np.mean(r*r)),'shift RMS',np.sqrt(np.mean(r.reshape(len(s['mu_shifts']),-1)**2,axis=1)), 'weights',w.round(5))
  else: errs.append(y[~keep]-S[~keep]@w-z[~keep]@gamma)
 E=np.asarray(errs); print('  held-rosette RMS pooled/worst',np.sqrt(np.mean(E**2)),np.sqrt(np.max(np.mean(E**2,axis=1))), 'bank RMS',np.sqrt(np.mean(E**2,axis=0)))
