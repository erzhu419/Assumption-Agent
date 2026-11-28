import sys, json, pickle, importlib.util, itertools
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
path='/tmp/hegel_continual_archive/beta_m100_c03_batch_08_solver.py'
spec=importlib.util.spec_from_file_location('archive08',path); ar=importlib.util.module_from_spec(spec); spec.loader.exec_module(ar)
with open('/holdout_task/public_batch.pkl','rb') as f: raw=pickle.load(f)[0]['study']
# Fit dynamically on the raw sweeps, then pass through the staged upstream transducer.
s=calibrate_upstream(raw)
y=s['y_obs_raw']; Z=s['Z_nuisance']; nx,ny=s['nx'],s['ny']; m=len(s['source_modes']); cap=s['u_box']; budget=s['total_budget']
indices=s['sensor_ij'] @ [1,nx]; rhs=s['source_modes'].transpose(0,2,1).reshape(m,-1).T

def responses(kinds):
 axes=[]
 for d,n,kind in zip('xy',(nx,ny),kinds):
  h=s['L'+d]/(n-1)
  a=ar._axis(s['kappa_'+d],s['vel_'+d],h,kind=='P')
  if kind=='R':
   a=a+diags(np.r_[s['robin_'+d]/h,np.zeros(n-2),s['robin_'+d]/h])
  axes.append(a)
 op=(kron(eye(ny),axes[0])+kron(axes[1],eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.vstack([splu(op+mu*eye(nx*ny,format='csc')).solve(rhs)[indices] for mu in s['mu_shifts']])

def fit(S, train, source_ridge=0., nuisance_ridge=0.):
 D=S[train]; z=Z[train]; obs=y[train]
 B=np.column_stack((D,obs)); B-=z@np.linalg.solve(z.T@z+nuisance_ridge*np.eye(z.shape[1]),z.T@B); Dp,yp=B[:,:-1],B[:,-1]
 w,nu=ar._box_simplex(Dp.T@Dp+source_ridge*np.eye(m),-Dp.T@yp,cap,budget)
 g=np.linalg.solve(z.T@z+nuisance_ridge*np.eye(z.shape[1]),z.T@(obs-D@w))
 return w,S@w+Z@g

def report(S):
 rows=np.arange(len(y)); sensor=rows%len(indices); shift=rows//len(indices)
 out=[]
 for scheme in (sensor,shift,sensor%4):
  errors=[]
  for k in np.unique(scheme):
   test=scheme==k
   w,pred=fit(S,~test); errors.append(np.sqrt(np.mean((pred[test]-y[test])**2)))
  out.append((float(np.sqrt(np.mean(np.square(errors)))),max(errors)))
 w,pred=fit(S,np.ones(len(y),bool)); fiterr=np.sqrt(np.mean((pred-y)**2))
 return fiterr,out,np.round(w,4)

if __name__=='__main__':
 print('law (x/y); P periodic, N reflecting, R Robin; full residual RMSE; (pooled,worst) LOSO / held-out shift / held-out rosette channel; weights')
 for kinds in itertools.product('PNR',repeat=2):
  full,scores,w=report(responses(kinds)); print(''.join(kinds),f'{full:.7g}',np.round(scores,7).tolist(),w)
