import importlib.util,pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
p='/tmp/hegel_continual_archive/dev_beta_m99_00_s00_solver.py'
spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; nx,ny=s['nx'],s['ny']; ns=nx*ny
pulses=item['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in pulses]).T; Y=np.array([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]).T
ij=s['sensor_ij']; pos=ij[:,0]+nx*ij[:,1]; src=s['source_modes'].transpose(0,2,1).reshape(-1,ns).T@W
axes=[]
for k,n in zip('xy',(nx,ny)):
 h=s['L'+k]/(n-1); kap=s['kappa_'+k]; axes.append((m._axis(kap,s['vel_'+k],h,s['robin_'+k]),kap[[0,-1]]/h**2,s['robin_'+k]/h))
V=diags(s['V_reaction_2d'].ravel(order='F')); I=eye(ns,format='csc')
def pred(t,wrap=False):
 mats=[]
 for axis,(base,k,r) in enumerate(axes):
  A=base.tolil(); n=A.shape[0]
  for side,(i,j) in enumerate(((0,1),(n-1,n-2))):
   a,b=t[4*axis+2*side:4*axis+2*side+2]; A[i,j]=-a*k[side]; A[i,i]=a*k[side]+b*r
   if wrap:
    c=t[8+2*axis+side]*k[side]; A[i,n-1-i]=-c; A[i,i]+=c
  mats.append(A.tocsc())
 A=(kron(eye(ny),mats[0])+kron(mats[1],eye(nx))+V).tocsc()
 return np.vstack([splu(A+mu*I).solve(src)[pos] for mu in s['mu_shifts']])
def report(label,t,wrap=False):
 d=(pred(t,wrap)-Y).reshape(len(s['mu_shifts']),len(pos),-1); print(label,'parameters',np.round(t,6),'rms',np.sqrt(np.mean(d*d)),'max by shift/pulse',np.max(abs(d),axis=1).round(7))
report('open',np.ones(8))
for wrap in (False,True):
 t0=np.r_[np.ones(8),np.zeros(4)] if wrap else np.ones(8)
 fit=least_squares(lambda t:(pred(t,wrap)-Y).ravel(),t0,bounds=(-0.05,12),gtol=1e-13,ftol=1e-12,xtol=1e-12)
 report('fitted wrap='+str(wrap),fit.x,wrap); print('sensitivity singular values',np.linalg.svd(fit.jac,compute_uv=False).round(6))
 for j in range(W.shape[1]):
  train=np.arange(W.shape[1])!=j
  q=least_squares(lambda t:(pred(t,wrap)-Y)[:,train].ravel(),t0,bounds=(-0.05,12),gtol=1e-12).x
  e=pred(q,wrap)[:,j]-Y[:,j]; print('held pulse',j,'max',max(abs(e)),'rms',np.sqrt(np.mean(e*e)))
