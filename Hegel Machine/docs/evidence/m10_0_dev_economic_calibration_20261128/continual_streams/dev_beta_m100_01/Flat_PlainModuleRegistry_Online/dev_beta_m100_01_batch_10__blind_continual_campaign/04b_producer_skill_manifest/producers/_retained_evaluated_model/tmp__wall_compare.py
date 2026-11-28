import sys, json, pickle, itertools, importlib.util
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('core','/tmp/hegel_continual_archive/dev_beta_m100_01_batch_09_solver.py'); core=importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
s=calibrate_upstream(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']); y=s['y_obs_raw']; t=s['aux_manifold_telemetry']; Z=s['Z_nuisance']; m=s['source_modes'].shape[0]
print('aux mean/std',t.mean(0),t.std(0), 'first four',t[:4]); print('sensor rows',s['sensor_ij'].tolist()); print('y quantiles',np.quantile(y,[0,.25,.5,.75,1]))
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for r in log:
 q=np.array(r['facility_telemetry_trace']); dif=np.sqrt(np.mean((t[:,:,None]-q[:,None,:])**2,axis=0)); print('log',r['completed_campaign_index'],'nearest channels',dif.argmin(1),np.round(dif.min(1),3))
def axis(k,v,h,r,mode):
 A=core._axis(k,v,h)
 if mode=='periodic': return A
 A=A.tolil(); A[0,:]=0; A[-1,:]=0
 for i,j in ((0,1),(len(k)-1,len(k)-2)):
  A[i,j]=-k[i]/h**2; A[i,i]=-A[i,j]+(r/h if mode=='robin' else 0)
 return A.tocsc()
def forward(x,y):
 nx,ny=s['nx'],s['ny']; X=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],x); Y=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],y)
 L=sparse.kron(sparse.eye(ny),X,format='csc')+sparse.kron(Y,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=sparse.eye(nx*ny,format='csc'); ix,iy=s['sensor_ij'].T; at=ix+nx*iy; rhs=s['source_modes'].transpose(0,2,1).reshape(m,-1).T
 return np.vstack([splu(L+mu*I).solve(rhs)[at] for mu in s['mu_shifts']])
def fit(S, mask):
 A=S[mask]; z=Z[mask]; yy=y[mask]; P=lambda b:b-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@b); w,_=core._qp(P(A),P(yy),s['source_ridge'],s['u_box'],s['total_budget']); gam=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@(yy-A@w)); return y-S@w-Z@gam
for x,b in itertools.product(('robin','sealed','periodic'),repeat=2):
 S=forward(x,b); row=np.arange(len(y)); res=fit(S,row>=0); groups=row//4%(len(s['sensor_ij'])//4); cv=np.concatenate([fit(S,groups!=j)[groups==j] for j in np.unique(groups)]); print(x,b,'train',np.sqrt(np.mean(res**2)),'CV',np.sqrt(np.mean(cv**2)),'worst rosette family',np.max(np.sqrt(np.mean(cv.reshape(len(np.unique(groups)),-1)**2,axis=1))))
