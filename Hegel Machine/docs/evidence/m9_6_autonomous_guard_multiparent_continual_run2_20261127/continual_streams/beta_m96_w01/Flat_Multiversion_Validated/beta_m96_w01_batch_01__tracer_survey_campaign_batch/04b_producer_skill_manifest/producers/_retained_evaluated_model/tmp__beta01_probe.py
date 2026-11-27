import importlib.util, pickle
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
p='/tmp/hegel_continual_archive/beta_m96_w01_s00_solver.py'
spec=importlib.util.spec_from_file_location('prior',p); prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
item=pickle.load(open('/holdout_task/public_beta_m96_w01_batch_01.pkl','rb'))[0]; s,g=item['study'],item['gt']; nx,ny=s['nx'],s['ny']; V=s['V_reaction_2d']; R=s['reaction_x'][:,None]+s['reaction_y'][None,:]
print('V range',V.min(),V.max(),'nonadditivity',np.max(np.abs(V-V[:,[0]]-V[[0],:]+V[0,0])),'R range',R.min(),R.max())
axes=[sparse.csc_matrix(prior._build_1d(s['L'+a],s['n'+a],s['kappa_'+a],s['vel_'+a],s['robin_'+a],np.zeros(s['n'+a]))) for a in 'xy']; I=sparse.eye(nx*ny,format='csc'); K=sparse.kron(axes[0],sparse.eye(ny),format='csc')+sparse.kron(sparse.eye(nx),axes[1],format='csc'); src=s['source_modes'].reshape(-1,nx*ny).T; ids=np.ravel_multi_index(s['sensor_ij'].T,(nx,ny))
z=s['Z_nuisance']; Q,d,_=np.linalg.svd(z,full_matrices=False); alpha=d*d/(d*d+s['nuisance_ridge']); proj=lambda x,a:x-Q@(a[:,None]*(Q.T@x))
y=prior._calibrate(s)[:,None]; a2=1-np.sqrt(1-alpha)
print('target raw/P/sqrt maxerr',*[np.max(np.abs(f[:,0]-g['r_y'])) for f in [y,proj(y,alpha),proj(y,a2)]])
for name,reaction in [('axis',R),('V',V),('axis+V',R+V)]:
 A=K+sparse.diags(reaction.ravel(),format='csc'); fields=[splu(A+mu*I).solve(src) for mu in s['mu_shifts']]; S=np.concatenate([f[ids] for f in fields]); target=g['S_tilde']; fielderr=[np.max(np.abs((f@g['weights_by_budget'][1]).reshape(nx,ny)-g['terminal_field_mid_budget'])) for f in fields]; print(name,'design raw/P/sqrt maxerr',*[np.max(np.abs(f-target)) for f in [S,proj(S,alpha),proj(S,a2)]],'terminal best shift index/error',np.argmin(fielderr),min(fielderr))
