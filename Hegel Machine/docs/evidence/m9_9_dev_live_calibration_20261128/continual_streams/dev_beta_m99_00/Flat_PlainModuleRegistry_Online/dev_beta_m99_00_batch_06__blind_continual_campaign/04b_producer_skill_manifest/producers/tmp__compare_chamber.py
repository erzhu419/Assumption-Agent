import sys, json, pickle, importlib.util, itertools
import numpy as np
from scipy.sparse import eye, diags, kron
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
p='/tmp/hegel_continual_archive/dev_beta_m99_00_batch_05_solver.py'
spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); s=calibrate_upstream(st,pay); dyn=calibrate_upstream(st)
y=s['y_obs_raw']; print('staged vs fitted calibration max delta',np.max(np.abs(y-dyn['y_obs_raw'])))
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; pos=ij[:,0]+nx*ij[:,1]; ns=len(pos); nr=len(y); Z=s['Z_nuisance']; ridge=s['nuisance_ridge']; sr=s['source_ridge']; cap=s['u_box']; total=s['total_budget']; rhs=np.column_stack([a.ravel(order='F') for a in s['source_modes']]); I=eye(nx*ny,format='csc')
def axis(key,n,kind):
 h=s['L'+key]/(n-1); a=old._axis(s['kappa_'+key],s['vel_'+key],h,kind=='P')
 if kind=='R':
  d=np.zeros(n); d[[0,-1]]=s['robin_'+key]/h; a=a+diags(d)
 return a

def fit(S,tr):
 a,b,yy=S[tr],Z[tr],y[tr]; pz=np.linalg.solve(b.T@b+ridge*np.eye(b.shape[1]),b.T); a0=a-b@(pz@a); y0=yy-b@(pz@yy); w,nu=old._box_simplex(a0.T@a0+sr*np.eye(a.shape[1]),-a0.T@y0,cap,total); return a0,y0,w,pz@(yy-a@w)

def rmse(r): return np.sqrt(np.mean(r*r))
idx=np.arange(nr); groups=np.tile(np.arange(ns)//4,nr//ns)
for x,b in itertools.product('RSP',repeat=2):
 L=(kron(eye(ny),axis('x',nx,x))+kron(axis('y',ny,b),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); S=np.vstack([splu(L+mu*I).solve(rhs)[pos] for mu in s['mu_shifts']]); a,y0,w,gamma=fit(S,np.ones(nr,dtype=bool)); scores=[]
 for ids in (idx//ns,groups,s['row_bank_id']):
  e=[]
  for g in np.unique(ids):
   tr=ids!=g; _,_,v,c=fit(S,tr); e.append(rmse((S@v+Z@c-y)[~tr]))
  scores.append((round(float(np.mean(e)),6),round(float(np.max(e)),6)))
 print(x+b,'all_rmse',round(rmse(y-S@w-Z@gamma),6),'J',round((np.linalg.norm(a@w-y0)**2+sr*(w@w))/2,7),'mean/max held-shift, rosette, bank',scores)
