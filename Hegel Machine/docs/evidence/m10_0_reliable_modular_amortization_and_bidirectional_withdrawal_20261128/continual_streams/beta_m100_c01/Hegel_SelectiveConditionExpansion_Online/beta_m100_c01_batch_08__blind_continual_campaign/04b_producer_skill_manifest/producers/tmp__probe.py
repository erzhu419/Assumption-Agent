import sys,pickle,numpy as np
from scipy.linalg import solve
sys.path.insert(0,'/holdout_task')
import _verified_upstream as up
from importlib.machinery import SourceFileLoader
arch=SourceFileLoader('arch','/tmp/hegel_continual_archive/beta_m100_c01_batch_02_solver.py').load_module()
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=d['study']; pulse=d['campaign_diagnostics']; cal=up.calibrate_upstream(dict(st,reference_tracer_pulses=pulse)); W=np.stack([p['w_pulse'] for p in pulse],axis=1); Y=np.stack([p['y_pulse_raw'] for p in cal['reference_tracer_pulses']],axis=-1)
nx,ny=st['nx'],st['ny']; ij=st['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; B=np.column_stack([s.ravel(order='F') for s in st['source_modes']])@W

def axis(a,kind):
 k=st['kappa_'+a]; v=st['vel_'+a]; n=len(k); h=st['L'+a]/(n-1); r=st['robin_'+a]; f=(k[1:]+k[:-1])/(2*h*h); u=np.zeros((n,n)); i=np.arange(1,n-1)
 u[i,i-1]=-f[:-1]-np.maximum(v[i],0)/h; u[i,i+1]=-f[1:]-np.maximum(-v[i],0)/h; u[i,i]=-u[i,i-1]-u[i,i+1]
 for j,l in [(0,1),(n-1,n-2)]:
  u[j,l]=-k[j]/h**2; u[j,j]=-u[j,l]+(r/h if kind=='robin' else 0)
 if kind=='archive': return arch._axis(k,v,h).toarray() if hasattr(arch._axis(k,v,h),'toarray') else arch._axis(k,v,h)
 return u

def model(ax,ay):
 A=np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(st['V_reaction_2d'].ravel(order='F'))
 return np.concatenate([solve(A+mu*np.eye(nx*ny),B)[idx] for mu in st['mu_shifts']])
if __name__=='__main__':
 print('cal pulse range',Y.min(),Y.max(),'rms',np.sqrt(np.mean(Y**2,axis=0)))
 for x in ('robin','reflect','archive'):
  for y in ('robin','reflect','archive'):
   err=model(axis('x',x),axis('y',y))-Y
   print(x,y,np.sqrt(np.mean(err**2,axis=0)),np.max(np.abs(err)))
 for a in 'xy':
  print(a,'kap',st['kappa_'+a][[0,1,-2,-1]],'vel',st['vel_'+a][[0,1,-2,-1]],'archive rows',axis(a,'archive')[[0,-1]])
