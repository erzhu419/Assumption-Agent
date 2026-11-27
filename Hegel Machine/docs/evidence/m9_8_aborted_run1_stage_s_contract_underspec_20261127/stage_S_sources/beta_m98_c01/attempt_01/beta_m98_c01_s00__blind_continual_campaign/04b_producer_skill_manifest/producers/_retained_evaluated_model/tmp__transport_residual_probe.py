import sys
sys.path.insert(0,'/tmp')
from transport_probe import *
from importlib.util import spec_from_file_location, module_from_spec
P=spec_from_file_location('solver','/holdout_task/batch_solver.py'); M=module_from_spec(P); P.loader.exec_module(M)
correct=M._calibration(s['sensor_calibration_sweeps']); Y=np.column_stack([correct(p['y_pulse_raw']) for p in pulses]); n=len(Y); nr=len(idx); rows=np.arange(n); masks=[np.broadcast_to(mask[:,None],Y.shape) for mask in [(rows%nr//4)%4==f for f in range(4)]+[rows//nr==f for f in range(len(mu))]]; masks.extend(np.broadcast_to(np.arange(Y.shape[1])==j,Y.shape) for j in range(Y.shape[1])); A=op(0,2,1,1)
def delta(j, which):
 k=s['kappa_'+j]; v=s['vel_'+j]; h=s['L'+j]/(s['n'+j]-1); D=one(k,np.zeros_like(v),h,0,2,1) if which=='k' else one(np.zeros_like(k),v,h,0,2,1)
 return (kron(eye(ny),D) if j=='x' else kron(D,eye(nx))).tocsc()
Ds={j+t:delta(j,t) for j in 'xy' for t in ['k','v']}; Ds['reaction']=diags(s['V_reaction_2d'].ravel(order='F')); a0=np.repeat([s['robin_x']*(nx-1)/s['Lx'],s['robin_y']*(ny-1)/s['Ly']],2)
for keys in [(),('xk','yk'),('xv','yv'),('reaction',),('xk','yk','xv','yv','reaction')]:
 def pred(q):
  D=A.copy()
  for key,b in zip(keys,q[4:]): D=D+b*Ds[key]
  return predict(D,q[:4])
 x0=np.r_[a0,np.zeros(len(keys))]
 def fit_mask(mask): return least_squares(lambda q:(pred(q)-Y)[mask],x0,gtol=1e-13,ftol=1e-10,xtol=1e-11).x
 q=fit_mask(np.ones(Y.shape,bool)); cv=np.array([np.sqrt(np.mean((pred(fit_mask(~mask))-Y)[mask]**2)) for mask in masks]); e=(pred(q)-Y).reshape(len(mu),nr,-1)
 print(keys,'fit',np.sqrt(np.mean(e**2)),'worst',cv.max(),'folds',np.round(cv*1e6,2),'params',np.round(q,7))
print('archive:',__import__('pathlib').Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').exists())
