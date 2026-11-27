import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import splu
A=runpy.run_path('/tmp/hegel_continual_archive/beta_m99_c01_s00_solver.py')
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']
nx,ny=s['nx'],s['ny']; dx,dy=s['Lx']/(nx-1),s['Ly']/(ny-1)
ax=A['_axis'](s['kappa_x'],s['vel_x'],dx,s['robin_x']).toarray(); ay=A['_axis'](s['kappa_y'],s['vel_y'],dy,s['robin_y']).toarray()
print('baseline wall rows',ax[0,:2],ax[-1,-2:],ay[0,:2],ay[-1,-2:]); print('wall velocities',s['vel_x'][[0,-1]],s['vel_y'][[0,-1]])
print('aux mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0))
R=s['V_reaction_2d'].ravel(order='F'); i,j=s['sensor_ij'].T; sensor=i+nx*j; ident=eye(nx*ny,format='csc')
z=s['Z_nuisance']; P=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
def calibrate(raw):
 st=dict(s,y_obs_raw=raw); return A['_calibrate'](st)
Y=np.column_stack([calibrate(q['y_pulse_raw']) for q in pulses]); W=np.column_stack([q['w_pulse'] for q in pulses]); rhs=s['source_modes'].reshape(len(W),-1,order='F').T@W
support=lambda n: ([0,0,0,n-1,n-1,n-1],[0,1,n-1,n-1,n-2,0])
def predict(p):
 x,y=ax.copy(),ay.copy(); x[support(nx)]+=p[:6]; y[support(ny)]+=p[6:]; L=(kron(eye(ny),csc_matrix(x))+kron(csc_matrix(y),eye(nx))+diags(R)).tocsc()
 return np.concatenate([splu(L+shift*ident).solve(rhs)[sensor] for shift in s['mu_shifts']])
models={'diagonal': [0,3,6,9], 'local_rows':[0,1,3,4,6,7,9,10], 'local_and_wrap':list(range(12))}
for name,indices in models.items():
 def expand(q):
  p=np.zeros(12); p[indices]=q; return p
 def residual(q): return predict(expand(q))-Y
 for test in [-1,0,1,2]:
  train=[k for k in range(3) if k!=test]; f=least_squares(lambda q: residual(q)[:,train].ravel(),np.zeros(len(indices)),max_nfev=120,gtol=1e-11); E=residual(f.x)
  print(name,'held',test,'rms per pulse',np.round(np.sqrt(np.mean(E**2,axis=0)),8),'p',np.round(expand(f.x),6))
