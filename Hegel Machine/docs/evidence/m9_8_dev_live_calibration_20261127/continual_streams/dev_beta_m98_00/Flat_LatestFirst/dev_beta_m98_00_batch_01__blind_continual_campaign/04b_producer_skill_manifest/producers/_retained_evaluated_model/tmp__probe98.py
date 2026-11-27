import pickle, importlib.util, json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; d=p['campaign_diagnostics']; sw=s['sensor_calibration_sweeps']; ref=sw['ref_rosette']; raw=sw['raw_rosette']; nx,ny=s['nx'],s['ny']; W=np.array([a['w_pulse'] for a in d]).T; Y=np.array([a['y_pulse_raw'] for a in d]).T
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_beta_m98_00_s00_solver.py'); prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
def basis(x, deg):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,idx],axis=1) for j in range(1,deg+1) for idx in combinations_with_replacement(range(4),j)])
print('cal ranges',ref.min(0),ref.max(0),'rows',ref[:3]); print('single sweeps',sw['single_bank_sweeps']); print('pulse weights',W.T)
for deg in (1,2,3):
 t=basis(ref,deg); a=np.linalg.lstsq(t,raw,rcond=None)[0]; print('degree',deg,'cal in-sample rmse',np.sqrt(np.mean((t@a-raw)**2,axis=0)), 'odd/even rmse', np.sqrt(np.mean((t[1::2]@np.linalg.lstsq(t[::2],raw[::2],rcond=None)[0]-raw[1::2])**2,axis=0))); print('coeff',a if deg<2 else np.max(abs(a[5:]),axis=0))
base_cal=np.linalg.lstsq(basis(ref,1),raw,rcond=None)[0]
axs=[prior._axis(s['kappa_'+ax],s['vel_'+ax],s['L'+ax]/(n-1),s['robin_'+ax]) for ax,n in [('x',nx),('y',ny)]]; dims=(nx,ny)
rows=s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]; rhs=np.column_stack([x.ravel(order='F') for x in s['source_modes']])@W; eye=sparse.eye(nx*ny,format='csc'); react=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def forward(q,rhs=rhs):
 mat=[]
 for j,n in enumerate(dims):
  a=axs[j].tolil(); t=q[j*6:j*6+6]; a[0,0]+=t[0]; a[0,1]+=t[1]; a[-1,-1]+=t[2]; a[-1,-2]+=t[3]; a[0,-1]+=t[4]; a[-1,0]+=t[5]; mat.append(a.tocsc())
 op=sparse.kron(sparse.eye(ny),mat[0],format='csc')+sparse.kron(mat[1],sparse.eye(nx),format='csc')+react
 return np.vstack([splu(op+mu*eye).solve(rhs)[rows] for mu in s['mu_shifts']])
def pred(q,fit=base_cal,deg=1):
 f=forward(q); return np.stack([(basis(t.reshape(-1,4),deg)@fit).ravel() for t in f.T],axis=1)
for deg in [1,2]:
 t=basis(ref,deg); fit=np.linalg.lstsq(t,raw,rcond=None)[0]
 for which in ['all','train0','train1']:
  ind=slice(None) if which=='all' else [int(which[-1])]; fun=lambda q:(pred(q,fit,deg)-Y)[:,ind].ravel(); sol=least_squares(fun,np.zeros(12),max_nfev=150,ftol=1e-10); err=pred(sol.x,fit,deg)-Y; print(deg,which,'baseline',np.sqrt(np.mean(fun(np.zeros(12))**2)),'fit',np.sqrt(np.mean(err**2,axis=0)),'q',np.round(sol.x,5),'sing',np.round(np.linalg.svd(sol.jac,compute_uv=False),8))
print('baseline rows',[(a.toarray()[[0,-1]][:,[0,1,-2,-1]]) for a in axs]); print('archive telemetry mean',[np.mean(a['facility_telemetry_trace'],axis=0) for a in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))])
