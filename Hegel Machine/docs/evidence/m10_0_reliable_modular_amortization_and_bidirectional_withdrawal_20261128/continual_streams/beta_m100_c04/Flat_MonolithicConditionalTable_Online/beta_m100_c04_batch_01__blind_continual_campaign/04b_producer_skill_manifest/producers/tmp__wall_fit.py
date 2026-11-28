import importlib.util, pickle, sys
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
p='/tmp/hegel_continual_archive/beta_m100_c04_s00_solver.py'
spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=d['study']; nx,ny=s['nx'],s['ny']; n=nx*ny
pulses=d['campaign_diagnostics']; w=np.column_stack([p['w_pulse'] for p in pulses]); y=np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); base=[]; scales=[]
for k in 'xy':
 h=s['L'+k]/(s['n'+k]-1); kap=s['kappa_'+k]; base.append(m._axis(kap,s['vel_'+k],h,s['robin_'+k]).tolil()); scales.extend(kap[[0,-1]]/h**2)
R=sparse.diags(s['V_reaction_2d'].ravel(order='F')); I=sparse.eye(n,format='csc'); rows=s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]; rhs=np.stack([z.ravel(order='F') for z in s['source_modes']],axis=1)@w

def predict(theta, offsets):
 A=[]
 for j,b in enumerate(base):
  a=b.copy(); N=b.shape[0]
  for side in range(2):
   for off,v in zip(offsets,theta[2*j+side]):
    row=0 if side==0 else N-1; col=off%N if side==0 else N-1-off%N
    a[row,col]+=v*scales[2*j+side]
  A.append(a.tocsc())
 L=(sparse.kron(sparse.eye(ny),A[0])+sparse.kron(A[1],sparse.eye(nx))+R).tocsc()
 return np.concatenate([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])
for offsets in [(0,1),(0,1,-1),(0,1,2),(0,1,2,-1),(0,1,2,-1,-2)]:
 def fit(train):
  z=least_squares(lambda x:(predict(x.reshape(4,-1),offsets)-y)[:,train].ravel(),np.zeros(4*len(offsets)),max_nfev=150,gtol=1e-12,xtol=1e-11,ftol=1e-11)
  return z.x.reshape(4,-1)
 theta=fit([0,1,2]); print('offsets',offsets,'fit RMSE',np.sqrt(np.mean((predict(theta,offsets)-y)**2)),'coeffs norm',np.round(theta,6),'coeffs raw',np.round(theta*np.array(scales)[:,None],6),flush=True)
 for i in range(3):
  r=(predict(fit([j for j in range(3) if j!=i]),offsets)-y)[:,i]; print(' held pulse',i,np.sqrt(np.mean(r*r)), 'max shift',max(np.sqrt(np.mean(r.reshape(3,-1)**2,axis=1))),flush=True)
print('axis ends [kappa, vel, robin/h]');
for k in 'xy': print(k,s['kappa_'+k][[0,-1]],s['vel_'+k][[0,-1]],s['robin_'+k]*(s['n'+k]-1)/s['L'+k]);
