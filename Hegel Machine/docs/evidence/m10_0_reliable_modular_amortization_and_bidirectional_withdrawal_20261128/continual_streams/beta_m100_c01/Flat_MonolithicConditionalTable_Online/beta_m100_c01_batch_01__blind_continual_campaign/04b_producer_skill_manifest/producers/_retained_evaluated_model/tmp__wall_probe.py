import sys,pickle,importlib.util,numpy as np
from scipy.sparse import csc_matrix, diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/beta_m100_c01_s00_solver.py'
spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; pulses=a['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; W=np.column_stack([d['w_pulse'] for d in pulses]); Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in pulses]); Z=s['Z_nuisance']; ij=s['sensor_ij']; slots=ij[:,0]+nx*ij[:,1]; rhs=np.column_stack([v.ravel(order='F') for v in s['source_modes']])@W
axes=[m._axis(s['kappa_'+t],s['vel_'+t],s['L'+t]/(n-1),s['robin_'+t]).toarray() for t,n in [('x',nx),('y',ny)]]
I=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def pred(p):
 X,A=[x.copy() for x in axes]; q=np.asarray(p).reshape(4,-1)
 for (M,b,j,k),v in zip([(X,0,1,-1),(X,-1,-2,0),(A,0,1,-1),(A,-1,-2,0)],q):
  M[b,b]+=v[0]; M[b,j]+=v[1];
  if len(v)==3: M[b,k]+=v[2]
 L=kron(eye(ny),csc_matrix(X),format='csc')+kron(csc_matrix(A),eye(nx),format='csc')+V
 return np.vstack([splu(L+u*I).solve(rhs)[slots] for u in s['mu_shifts']])
def res(p,proj=False):
 R=pred(p)-Y
 if proj: R=R-Z@np.linalg.lstsq(Z,R,rcond=None)[0]
 return R
print('baseline pulse rmse:',np.sqrt(np.mean(res(np.zeros(8))**2,axis=0)), 'nuisance projected:',np.sqrt(np.mean(res(np.zeros(8),True)**2,axis=0)))
for t,M in zip('xy',axes): print('axis scales',t,'rows',M[0,:2],M[-1,-2:],'v/h',s['vel_'+t][[0,-1]]/(s['L'+t]/(len(M)-1)))
for n in [8,12]:
 for proj in [False,True]:
  for leave in [-1,0,1,2]:
   train=np.arange(3)!=leave
   fit=least_squares(lambda p:res(p,proj)[:,train].ravel(),np.zeros(n),max_nfev=150,ftol=1e-11,xtol=1e-11,gtol=1e-12)
   print(n,proj,leave,'params',np.round(fit.x,6),'rmse',np.sqrt(np.mean(res(fit.x,proj)**2,axis=0)))
