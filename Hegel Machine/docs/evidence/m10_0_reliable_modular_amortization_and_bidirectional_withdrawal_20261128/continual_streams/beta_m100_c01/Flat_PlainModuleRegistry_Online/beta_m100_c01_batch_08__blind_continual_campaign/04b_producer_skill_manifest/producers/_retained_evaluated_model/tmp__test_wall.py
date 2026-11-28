import sys,pickle,itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.linalg import solve
from _verified_upstream import calibrate_upstream
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(p['study'],reference_tracer_pulses=p['campaign_diagnostics']))
Y=np.array([d['y_pulse_raw'] for d in s['reference_tracer_pulses']]).T
W=np.array([d['w_pulse'] for d in s['reference_tracer_pulses']]).T
nx,ny=s['nx'],s['ny']; ids=s['sensor_ij'] @ [1,nx]; I=np.eye(nx*ny)
M=s['source_modes'].transpose(0,2,1).reshape(len(W),-1).T
B=M@W
Z=s['Z_nuisance']
def ax(a,kind):
 k,v=s['kappa_'+a],s['vel_'+a]; n=len(k); h=s['L'+a]/(n-1); A=np.zeros((n,n))
 for i in range(n):
  if i in (0,n-1) and kind not in ('periodic','periodic-robin'):
   j=1 if i==0 else n-2
   d=k[i]/h**2; A[i,j]=-d; A[i,i]=d+(s['robin_'+a]/h if kind=='robin' else 0)
  else:
   j=(i-1)%n; l=(i+1)%n
   A[i,j]=-(k[i]+k[j])/(2*h**2)-max(v[i],0)/h; A[i,l]=-(k[i]+k[l])/(2*h**2)-max(-v[i],0)/h
   A[i,i]=-A[i].sum()+ (s['robin_'+a]/h if i in (0,n-1) and kind=='periodic-robin' else 0)
 return A
def forward(Ax,Ay,B=B):
 L=np.kron(np.eye(ny),Ax)+np.kron(Ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
 return np.concatenate([solve(L+mu*I,B)[ids] for mu in s['mu_shifts']])
if __name__=='__main__':
 scores=[]
 for x,y in itertools.product(('robin','closed','periodic','periodic-robin'),repeat=2):
  r=forward(ax('x',x),ax('y',y))-Y
  scores.append((np.sqrt(np.mean(r*r)),x,y,np.sqrt(np.mean(r*r,axis=0)).round(7),np.sqrt(np.mean((r-Z@np.linalg.lstsq(Z,r,rcond=None)[0])**2))))
 print('raw RMSE, x, y, pulse RMSEs, nuisance-profiled RMSE')
 for r in sorted(scores): print(*r)
