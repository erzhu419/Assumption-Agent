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
from scipy.optimize import minimize
from functools import lru_cache
closed=[a._axis(s['kappa_'+ax],s['vel_'+ax],s['L'+ax]/(n-1),periodic=False) for ax,n in zip('xy',(nx,ny))]
wrapped=[a._axis(s['kappa_'+ax],s['vel_'+ax],s['L'+ax]/(n-1),periodic=True) for ax,n in zip('xy',(nx,ny))]
@lru_cache(maxsize=256)
def seam(x,z):
 A=kron(eye(ny),closed[0]+x*(wrapped[0]-closed[0]),format='csc')+kron(closed[1]+z*(wrapped[1]-closed[1]),eye(nx),format='csc')+R
 return np.vstack([splu(A+mu*I).solve(rhs)[slots] for mu in s['mu_shifts']])
# Estimate seam strengths only inside training folds. A reproducible gain would
# support replacing the discrete law; pooled fit alone cannot justify it.
def train_error(p,tr):
 S=seam(*p); return rms(fit(S,tr,rid)[tr])**2
fixed={'CC':seam(0,0),'CP':seam(0,1),'PC':seam(1,0),'PP':seam(1,1)}
for axis,n in enumerate((nx,ny)):
 for label,A in [('closed',closed[axis]),('periodic',wrapped[axis])]:
  D=A.toarray(); assert np.max(np.abs(D.sum(axis=1)))<1e-10
  assert np.max(D-np.diag(np.diag(D)))<=0
  assert bool(D[0,-1]!=0 and D[-1,0]!=0)==(label=='periodic')
for label,group in [('sensor',folds),('shift',[np.arange(len(y))//len(ij)==k for k in range(len(s['mu_shifts']))]),('bank',[s['row_bank_id']==k for k in np.unique(s['row_bank_id'])])]:
 e={k:np.zeros(len(y)) for k in (*fixed,'nested discrete','nested partial')}; selected=[]; partial=[]
 for te in group:
  tr=~te
  scores={k:rms(fit(S,tr,rid)[tr]) for k,S in fixed.items()}
  best=min(scores,key=scores.get); selected.append(best)
  best_p={'CC':(0,0),'CP':(0,1),'PC':(1,0),'PP':(1,1)}[best]
  p=min((minimize(train_error,start,args=(tr,),method='L-BFGS-B',bounds=[(0,1),(0,1)],options={'gtol':1e-10,'ftol':1e-14}) for start in (best_p,(0.5,0.5))),key=lambda t:t.fun).x
  partial.append(p.round(2).tolist())
  for k,S in {**fixed,'nested discrete':fixed[best],'nested partial':seam(*p)}.items(): e[k][te]=fit(S,tr,rid)[te]
 print(label,'(overall,worst fold)',{k:[round(rms(v),7),round(max(rms(v[t]) for t in group),7)] for k,v in e.items()})
 print(label,'training-only choices',selected,'partial',partial)
 if label=='bank': print('PC held-out bank biases', [round(np.mean(e['PC'][t]),7) for t in group])
