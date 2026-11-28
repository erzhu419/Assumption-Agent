import sys,json,pickle,itertools,importlib.util
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('arch','/tmp/hegel_continual_archive/beta_m100_c01_batch_08_solver.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
s=calibrate_upstream(st,json.load(open('/holdout_task/_upstream_param_payload.json')))
print('calibrated additions',set(s)-set(st))
y=s['y_obs_cal'] if 'y_obs_cal' in s else s['y_obs_raw']; y=np.asarray(y).ravel()
Z=s['Z_nuisance']; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; slots=ij[:,0]+nx*ij[:,1]; rid=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']
R=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); loc=np.tile(ij,(len(s['mu_shifts']),1)); strips=(loc[:,0]==0)|(loc[:,0]==nx-1)|(loc[:,1]==0)|(loc[:,1]==ny-1)
def design(modes):
 ops=[]
 for axis,n,mode in zip('xy',(nx,ny),modes):
  h=s['L'+axis]/(n-1); A=a._axis(s['kappa_'+axis],s['vel_'+axis],h,periodic=mode=='P')
  if mode=='R':
   d=np.zeros(n); d[[0,-1]]=s['robin_'+axis]/h; A=A+diags(d,format='csc')
  ops.append(A)
 A=kron(eye(ny),ops[0],format='csc')+kron(ops[1],eye(nx),format='csc')+R
 return np.vstack([splu(A+mu*I).solve(rhs)[slots] for mu in s['mu_shifts']])
def fit(S,tr,penalty):
 X=S[tr]; N=Z[tr]; yy=y[tr]; P=np.linalg.solve(N.T@N+s['nuisance_ridge']*np.eye(N.shape[1]),N.T); F=X-N@(P@X); q=yy-N@(P@yy)
 w,_=a._box_simplex(F.T@F+penalty*np.eye(X.shape[1]),-F.T@q,cap,budget)
 return y-S@w-Z@(P@(yy-X@w))
def rms(x): return np.sqrt(np.mean(x*x)) if x.size else np.nan
folds=[]
for i,j in np.unique(ij,axis=0): folds.append((loc[:,0]==i)&(loc[:,1]==j))
for m in itertools.product('RCP',repeat=2):
 S=design(m); es=[]
 for penalty in (rid,0.0):
  e=np.zeros(len(y)); full=fit(S,np.ones(len(y),dtype=bool),penalty)
  for te in folds: e[te]=fit(S,~te,penalty)[te]
  es.append(np.round([rms(full),rms(e),rms(e[strips]),max(rms(e[te]) for te in folds)],7))
 print(''.join(m),'[full, held-out, wall, worst-sensor] with/without source ridge',*es)
