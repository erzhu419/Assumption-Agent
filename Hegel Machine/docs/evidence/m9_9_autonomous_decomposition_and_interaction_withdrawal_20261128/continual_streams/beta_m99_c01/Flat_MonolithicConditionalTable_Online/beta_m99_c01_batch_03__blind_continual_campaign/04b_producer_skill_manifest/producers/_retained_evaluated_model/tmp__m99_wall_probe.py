import sys,pickle,importlib.util,itertools
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import root
from m99_cal_probe import design,fit
p=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m99_c01_batch_02_solver.py'); prior=importlib.util.module_from_spec(p);p.loader.exec_module(prior)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['sensor_calibration_sweeps']['ref_rosette']; raw=s['sensor_calibration_sweeps']['raw_rosette']; a=design(x,'pair2'); b=fit(a,raw); good=np.max(abs(a@b-raw),axis=1)<1e-3; a=np.delete(a,[5,6,7,8],axis=1); b=np.linalg.lstsq(a[good],raw[good],rcond=None)[0]
def cal(v): return np.delete(design(v,'pair2'),[5,6,7,8],axis=1)@b
y=[]
for target in s['y_obs_raw'].reshape(-1,4):
 sol=root(lambda v:cal(v[None])[0]-target,np.linalg.solve(b[1:5].T,target-b[0])); assert np.max(abs(sol.fun))<1e-8; y.extend(sol.x)
y=np.array(y); print('clean sweeps',good.sum(),'worst inlier',np.max(abs(a[good]@b-raw[good])),'calibrated range',y.min(),y.max())
shape=(s['nx'],s['ny']); nn=np.prod(shape); ix,jx=s['sensor_ij'].T; idx=ix+shape[0]*jx; modes=s['source_modes']; rhs=np.stack([v.ravel(order='F') for v in modes],axis=1)
def axis(axis,n,wall):
 k=s['kappa_'+axis]; v=s['vel_'+axis]; h=s['L'+axis]/(n-1); robin=s['robin_'+axis]
 if wall=='P': return prior._axis(k,v,h)
 out=prior._axis(k,v,h).tolil()
 for i,j in [(0,1),(n-1,n-2)]:
  out[i,:]=0; out[i,j]=-k[i]/h**2; out[i,i]=k[i]/h**2+(robin/h if wall=='R' else 0)
 return out.tocsc()
def forward(walls):
 ax,ay=[axis(d,n,w) for d,n,w in zip('xy',shape,walls)]; op=(kron(eye(shape[1]),ax)+kron(ay,eye(shape[0]))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); mats=[splu(op+mu*eye(nn,format='csc')).solve(rhs) for mu in s['mu_shifts']]; return np.vstack([m[idx] for m in mats])
z=s['Z_nuisance']; rho=s['nuisance_ridge']; cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']
def estimate(S,tr,ridge):
 Z=z[tr]; D=S[tr]; Y=y[tr]; p=np.linalg.solve(Z.T@Z+rho*np.eye(Z.shape[1]),Z.T); T=D-Z@(p@D); r=Y-Z@(p@Y); w,nu=prior._box_simplex(T.T@T+ridge*np.eye(len(modes)),-T.T@r,cap,budget); return w,p@(Y-D@w)
rows=np.arange(len(y)); masks=[(rows%len(idx))//4==k for k in range(len(idx)//4)]; shifts=[rows//len(idx)==k for k in range(len(s['mu_shifts']))]; allrows=rows>=0
for walls in itertools.product('RNP',repeat=2):
 S=forward(walls); outputs=[]
 for r in (0,ridge):
  w,g=estimate(S,allrows,r); rmse=lambda e:np.sqrt(np.mean(e**2)); cv=[]
  for group in (masks,shifts):
   errs=[]
   for te in group:
    wt,gt=estimate(S,~te,r); errs.append(rmse(S[te]@wt+z[te]@gt-y[te]))
   cv.append(max(errs))
  outputs.append(np.round([rmse(S@w+z@g-y),*cv],6))
 print(''.join(walls),*outputs)
