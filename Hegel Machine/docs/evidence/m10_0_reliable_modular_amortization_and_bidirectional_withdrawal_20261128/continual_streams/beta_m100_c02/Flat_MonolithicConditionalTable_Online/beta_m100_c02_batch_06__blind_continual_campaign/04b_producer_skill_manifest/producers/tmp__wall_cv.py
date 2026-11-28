import sys,pickle
sys.path.insert(0,'/tmp')
import numpy as np
from fit_cal import fit_cal
from scipy.optimize import root, least_squares
from scipy import sparse
from scipy.sparse.linalg import splu
from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
base=Path('/tmp/hegel_continual_archive')
def load(n):
 q=spec_from_file_location('m',base/f'beta_m100_c02_batch_{n}_solver.py');m=module_from_spec(q);q.loader.exec_module(m);return m
m,l,p=load('05'),load('03'),load('04')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; mm=fit_cal(s['sensor_calibration_sweeps'],(),'poly',12)
def f(x):return np.array([mm[j](x[j:j+1],x[(j+2)%4:(j+2)%4+1])[0] for j in range(4)])
y=np.array([root(lambda x:f(x)-t,np.zeros(4),tol=1e-11).x for t in s['y_obs_raw'].reshape(-1,4)]).ravel(); Z=s['Z_nuisance']; cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']
robin=np.repeat([s['robin_x'],s['robin_y']],2)
def response(walls, periodic=(False,False)):
 nx,ny=s['nx'],s['ny']; axes=[]
 for i,k in enumerate('xy'):
  args=(s['kappa_'+k],s['vel_'+k],s['L'+k]/((nx,ny)[i]-1)); axes.append(p.axis_operator(*args,periodic=True) if periodic[i] else l.axis_operator(*args,walls[2*i:2*i+2]))
 A=sparse.kron(sparse.eye(ny),axes[0],format='csc')+sparse.kron(axes[1],sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); src=np.column_stack([a.ravel(order='F') for a in s['source_modes']]); obs=s['sensor_ij']@np.array([1,nx]); S=[]
 for u in s['mu_shifts']:
  U=splu(A+u*sparse.eye(nx*ny,format='csc')).solve(src); S.append(U[obs]);
 return np.vstack(S),U
S,_=response(np.zeros(4)); SS,_=l.chamber_response(s,np.zeros(4));print('independent forward comparison',abs(S-SS).max())
def res(S, train, test, reg=ridge):
 z=Z[train]; sy=np.c_[S[train],y[train]]; sy-=z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@sy); a,b=sy[:,:-1],sy[:,-1]; w,nu=m.box_simplex(a.T@a+reg*np.eye(a.shape[1]),a.T@b,cap,budget); gamma=np.linalg.lstsq(z,y[train]-S[train]@w,rcond=None)[0];return y[test]-S[test]@w-Z[test]@gamma
ix=np.arange(len(y)); folds=[ix[np.tile(np.arange(8)%4==k,3).repeat(4)] for k in range(4)]
for per in [(False,False),(True,False),(False,True),(True,True)]:
 for kx in ([0.] if per[0] else [0.,1.]):
  for ky in ([0.] if per[1] else [0.,1.]):
   S,_=response(robin*np.repeat([kx,ky],2),per); scores=[np.sqrt(np.mean(res(S,np.setdiff1d(ix,t),t)**2)) for t in folds]; horizons=[np.sqrt(np.mean(res(S,ix[ix//32!=k],ix[ix//32==k])**2)) for k in range(3)]; print(per,kx,ky,'fit',np.linalg.norm(res(S,ix,ix))/len(ix)**.5,'rosette folds',np.round(scores,5),'shift folds',np.round(horizons,5))
for reg in [0.,ridge]:
 q=least_squares(lambda q:res(response(q)[0],ix,ix,reg),np.full(4,.02),bounds=(-.1,2.));print('profile four losses',reg,q.x,'RMS',np.std(q.fun),'sealed',np.std(res(response(np.zeros(4))[0],ix,ix,reg)))
