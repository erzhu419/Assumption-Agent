import importlib.util,pickle
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
path='/tmp/hegel_continual_archive/beta_m98_v2_c02_batch_02_solver.py'
spec=importlib.util.spec_from_file_location('arch',path); arch=importlib.util.module_from_spec(spec); spec.loader.exec_module(arch)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; nx,ny=s['nx'],s['ny']; shifts=s['mu_shifts']; pulses=item['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in pulses]).T
x=s['sensor_calibration_sweeps']['ref_rosette']; y=s['sensor_calibration_sweeps']['raw_rosette']; D=np.array([np.linalg.lstsq(np.c_[np.ones(len(x)),x[:,j]],y[:,j],rcond=None)[0] for j in range(4)])
ycal=np.array([((p['y_pulse_raw'].reshape(-1,4)-D[:,0])/D[:,1]).ravel() for p in pulses]).T
mode=s['source_modes'].reshape(-1,nx*ny,order='F').T@W
ix,iy=s['sensor_ij'].T; rows=ix+nx*iy
axes=[]
for a in 'xy':
 k=s['kappa_'+a]; h=s['L'+a]/(len(k)-1); axes.append((arch._axis(k,s['vel_'+a],h,0),h,s['robin_'+a],k))
I=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def pred(q):
 op=[]
 for (A,h,r,k),(leakL,leakR,deltaL,deltaR,wrapL,wrapR) in zip(axes,np.reshape(q,(2,6))):
  B=A.tolil(); B[0,0]+=leakL*r/h+deltaL*k[0]/h**2+wrapL*k[0]/h**2; B[-1,-1]+=leakR*r/h+deltaR*k[-1]/h**2+wrapR*k[-1]/h**2; B[0,1]-=deltaL*k[0]/h**2; B[-1,-2]-=deltaR*k[-1]/h**2; B[0,-1]-=wrapL*k[0]/h**2; B[-1,0]-=wrapR*k[-1]/h**2; op.append(B.tocsc())
 L=sparse.kron(sparse.eye(ny),op[0],format='csc')+sparse.kron(op[1],sparse.eye(nx),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(mode)[rows] for mu in shifts])
def unpack(b,kind):
 if kind=='axis_leak': return np.c_[b,b,np.zeros((2,4))]
 if kind=='leak': return np.c_[b.reshape(2,2),np.zeros((2,4))]
 if kind=='leak_diff': return np.c_[b.reshape(2,4),np.zeros((2,2))]
 return b.reshape(2,6)
for kind,n in [('axis_leak',2),('leak',4),('leak_diff',8),('full',12)]:
 for use in ([0,1],[0],[1]):
  def fun(b): return (pred(unpack(b,kind))-ycal)[:,use].ravel()
  fit=least_squares(fun,np.zeros(n),max_nfev=120,gtol=1e-12,ftol=1e-12,xtol=1e-12)
  err=pred(unpack(fit.x,kind))-ycal
  print(kind,use,'q',np.round(unpack(fit.x,kind),6).tolist(),'pulse rms',np.sqrt(np.mean(err**2,axis=0)),'shift rms',np.round(np.sqrt(np.mean(err.reshape(len(shifts),-1,2)**2,axis=1)),8).tolist(),'min jac sv',np.linalg.svd(fit.jac,compute_uv=False)[-1])
