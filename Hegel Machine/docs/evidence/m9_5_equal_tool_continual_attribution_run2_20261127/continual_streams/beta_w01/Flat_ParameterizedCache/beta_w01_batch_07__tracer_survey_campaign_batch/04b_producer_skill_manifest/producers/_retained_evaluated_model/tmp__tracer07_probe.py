import numpy as np, pickle, importlib.util
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_w01_batch_06_solver.py'); p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
s=pickle.load(open('/holdout_task/public_beta_w01_batch_07.pkl','rb'))[0]['study']; v=s['V_reaction_2d']; vx=v.mean(axis=1); vy=v.mean(axis=0)-v.mean(); print('additive residual',np.linalg.norm(v-vx[:,None]-vy)/np.linalg.norm(v)); print('reaction ranges',[(np.min(a),np.max(a)) for a in (v,vx,vy,s['reaction_x'],s['reaction_y'])]); print('shifts',s['mu_shifts'])
nx,ny=v.shape; m=len(s['source_modes']); ix,iy=s['sensor_ij'].T
for mode in ('V_only','axis_only','sum'):
 ax,by=[p._build_1d(s['L'+a],n,s['kappa_'+a],s['vel_'+a],s['robin_'+a],s['reaction_'+a] if mode!='V_only' else np.zeros(n)) for a,n in [('x',nx),('y',ny)]]
 op=kron(ax,eye(ny),format='csc')+kron(eye(nx),by,format='csc')
 if mode!='axis_only': op+=diags(v.ravel(),format='csc')
 S=np.vstack([splu(op+mu*eye(nx*ny,format='csc')).solve(s['source_modes'].reshape(m,-1).T).reshape(nx,ny,m)[ix,iy] for mu in s['mu_shifts']]); Z=s['Z_nuisance']; R=s['y_obs_raw']; B=s['row_bank_id']; D=np.column_stack((S,Z)); sw=s['sensor_calibration_sweeps']; sb=sw['bank_id']; x=sw['ref_signal']; y=sw['raw_readout']; known=np.array([np.polyfit(x[sb==b],y[sb==b],1) for b in (0,1)])
 def params(q): return np.vstack((known,np.asarray(q[-4:]).reshape(2,2)))
 def err(q):
  cal=params(q); return cal[B,0]*(D@q[:-4])+cal[B,1]-R
 def fit(train):
  q=np.r_[np.linalg.lstsq(D[train],((R-known[np.minimum(B,1),1])/known[np.minimum(B,1),0])[train],rcond=None)[0],1.,0.,1.,0.]; return least_squares(lambda q:err(q)[train],q,ftol=1e-12,gtol=1e-12,xtol=1e-12).x
 q=fit(np.ones(len(R),bool)); cal=params(q); rm=lambda a:np.sqrt(np.mean(a**2)); cv=np.empty_like(R)
 for sensor in range(len(ix)):
  test=np.arange(len(R))%len(ix)==sensor; cv[test]=err(fit(~test))[test]
 print(mode,'sensor design rms',rm(S),'raw RMSE',rm(err(q)),'LOSO by bank',[rm(cv[B==b]) for b in range(4)],'cal',cal.tolist(),'weight range sum',q[:m].min(),q[:m].max(),q[:m].sum(),'sweep residuals',np.round((y-cal[sb,0]*x-cal[sb,1])[-6:],8))
