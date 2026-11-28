import sys,pickle,itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.sparse import coo_matrix,diags,eye,kron
from scipy.sparse.linalg import splu
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
Y=np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]); W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
nx,ny=s['nx'],s['ny']; modes=s['source_modes']; rhs=modes.transpose(0,2,1).reshape(len(modes),-1).T@W
idx=s['sensor_ij']@np.array([1,nx]); I=eye(nx*ny,format='csc')
def axis(d,law):
 k=s['kappa_'+d]; v=s['vel_'+d]; n=len(k); h=s['L'+d]/(n-1); i=np.arange(n); j=(i-1)%n; l=(i+1)%n
 lo=(k+k[j])/(2*h*h)+np.maximum(v,0)/h; hi=(k+k[l])/(2*h*h)+np.maximum(-v,0)/h; leak=np.zeros(n)
 if law != 'periodic':
  lo[0]=hi[-1]=0; hi[0]=k[0]/h**2; lo[-1]=k[-1]/h**2
  if law=='robin': leak[[0,-1]]=s['robin_'+d]/h
 return coo_matrix((np.r_[lo+hi+leak,-lo,-hi],(np.tile(i,3),np.r_[i,j,l])),shape=(n,n)).tocsc()
def predict(ax,ay):
 A=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.vstack([splu(A+mu*I).solve(rhs)[idx] for mu in s['mu_shifts']])
print('calibrated pulse RMS',np.sqrt(np.mean(Y**2,axis=0)))
for x,y in itertools.product(['robin','sealed','periodic'],repeat=2):
 E=predict(axis('x',x),axis('y',y))-Y; R=E-Q@(Q.T@E)
 print(x,y,'pulse RMSE',np.sqrt(np.mean(E**2,axis=0)).round(8),'profiled',np.sqrt(np.mean(R**2,axis=0)).round(8),'shift RMSE',np.sqrt(np.mean(E.reshape(3,-1)**2,axis=1)).round(8))
