import pickle,numpy as np,importlib.util
from scipy.optimize import least_squares
from scipy import sparse
from scipy.sparse.linalg import splu
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m100_c02_s00_solver.py'); a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];s=i['study']; c=s['sensor_calibration_sweeps']; ref=c['ref_rosette']; raw=c['raw_rosette']; bank=c['single_bank_sweeps'];
for j in range(4):
 m=bank['bank_id']==j; print('bank',j,np.polynomial.polynomial.polyfit(bank['ref_signal'][m],bank['raw_readout'][m],1))
f=np.c_[np.ones(len(ref)),ref]; aff=np.linalg.lstsq(f,raw,rcond=None)[0]; print('affine max/RMS',np.max(abs(f@aff-raw)),np.sqrt(np.mean((f@aff-raw)**2)),aff)
Y=np.array([a.calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in i['campaign_diagnostics']]).T
W=np.array([p['w_pulse'] for p in i['campaign_diagnostics']]).T
nx,ny=s['nx'],s['ny']; I=sparse.eye(nx*ny,format='csc'); ij=s['sensor_ij']; sensor=ij[:,0]+nx*ij[:,1]; R=s['source_modes'].reshape(-1,nx*ny,order='F').T@W
axes=[a.axis_operator(s['kappa_'+d],s['vel_'+d],s['L'+d]/(s['n'+d]-1),s['robin_'+d]).toarray() for d in 'xy'];
def predict(p,kind):
 ax=[]
 for A, pars in zip(axes,p.reshape(4,-1).reshape(2,2,-1)):
  B=A.copy()
  for k,(row,d) in enumerate([(0,1),(-1,-1)]):
   for coeff,tag in zip(pars[k],kind):
    col={'d':row,'n':row+d,'w':-1 if row==0 else 0,'s':row+2*d}[tag]; B[row,col]+=coeff
  ax.append(sparse.csc_matrix(B))
 L=sparse.kron(sparse.eye(ny),ax[0],format='csc')+sparse.kron(ax[1],sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
 return np.concatenate([splu(L+mu*I).solve(R)[sensor] for mu in s['mu_shifts']])
err=lambda x: np.sqrt(np.mean(x*x,axis=0))
print('baseline RMS per shift/pulse', [err(b) for b in (predict(np.zeros(4),'d')-Y).reshape(3,32,3)])
for kind in ('d','dn','dw','dnw','dns','dnws'):
 p0=np.zeros(4*len(kind)); results=[]
 for held in (None,0,1,2):
  use=[j for j in range(3) if j!=held]; fit=least_squares(lambda p:(predict(p,kind)-Y)[:,use].ravel(),p0,max_nfev=110,gtol=1e-13,ftol=1e-12,xtol=1e-12); e=err(predict(fit.x,kind)-Y); results.append(e if held is None else e[held]);
  if held is None: print(kind,'deltas',fit.x.reshape(4,-1).round(5),'jac cond',np.linalg.cond(fit.jac), flush=True)
 print(kind,'in/held RMS',results,flush=True)
print('baseline edge rows',[(A[0,:3],A[-1,-3:]) for A in axes]);print('telemetry study first+mean',s['aux_manifold_telemetry'][:3],s['aux_manifold_telemetry'].mean(0))
