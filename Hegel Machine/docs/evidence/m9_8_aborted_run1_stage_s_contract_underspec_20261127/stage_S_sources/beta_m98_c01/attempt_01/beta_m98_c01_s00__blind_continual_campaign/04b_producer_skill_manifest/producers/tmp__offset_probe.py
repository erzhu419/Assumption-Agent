import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fix=m._calibration(s['sensor_calibration_sweeps']); R,F,a0=m._chamber(s); pulses=s['reference_tracer_pulses']; B=F@np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([fix(p['y_pulse_raw']) for p in pulses]); ns=len(s['sensor_ij']); rows=np.arange(len(Y)); folds=[(rows%ns//4)%4==j for j in range(4)]+[rows//ns==j for j in range(len(s['mu_shifts']))]; masks=[np.broadcast_to(f[:,None],Y.shape) for f in folds]+[np.broadcast_to(np.arange(Y.shape[1])==j,Y.shape) for j in range(Y.shape[1])]
for label,G in [('axis',np.repeat(np.eye(2),2,axis=0)),('split',np.eye(4))]:
 q0=np.linalg.lstsq(G,a0,rcond=None)[0]
 for shift in [False,True]:
  x0=np.r_[q0,0.] if shift else q0
  def pred(q): return R(G@(q[:-1] if shift else q),B,q[-1] if shift else 0.)[0]
  def est(mask): return least_squares(lambda q:(pred(q)-Y)[mask],x0,gtol=1e-13,ftol=1e-10,xtol=1e-11).x
  q=est(np.ones(Y.shape,bool)); cv=[np.sqrt(np.mean((pred(est(~mask))-Y)[mask]**2)) for mask in masks]; print(label,shift,'params',q,'all',np.sqrt(np.mean((pred(q)-Y)**2)),'cv',np.round(cv,8))
