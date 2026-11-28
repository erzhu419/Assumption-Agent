import sys, json, pickle, numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
sys.path[:0]=['/holdout_task','/tmp/hegel_continual_archive']
import beta_m100_c03_batch_01_solver as arc
from _verified_upstream import calibrate_upstream
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); ds=p['campaign_diagnostics']
W=np.array([d['w_pulse'] for d in ds]).T
Y=np.array([calibrate_upstream({**s,'y_obs_raw':d['y_pulse_raw']},pay)['y_obs_raw'] for d in ds]).T
Z=s['Z_nuisance']; base=[arc._axis(s['kappa_'+d],s['vel_'+d],s['L'+d]/(s['n'+d]-1)) for d in 'xy']
def axes(k):
 out=[]
 for j,d in enumerate('xy'):
  axis=base[j].tolil(); n=axis.shape[0]; h=s['L'+d]/(n-1); kap=s['kappa_'+d]; rob=s['robin_'+d];
  for i,(at,nei,far) in enumerate(((0,1,n-1),(n-1,n-2,0))):
   m,l,c=k[j,i]; v=m*kap[at]/h**2; q=c*0.5*(kap[0]+kap[-1])/h**2
   axis[at,nei]=-v; axis[at,far]=-q; axis[at,at]=v+l*rob/h+q
  out.append(axis.tocsc())
 return out
def eval(k):return arc._responses(s,axes(k))[0]@W-Y
def fit(name,kind,hold=None):
 train=np.ones(Y.shape,bool)
 if isinstance(hold,int):train[:,hold]=False
 if isinstance(hold,np.ndarray):train[hold]=False
 def expand(x):
  if kind=='shared': return np.broadcast_to(np.r_[x,0],(2,2,3))
  if kind=='axis': return np.broadcast_to(np.column_stack([x.reshape(2,2),np.zeros(2)])[:,None,:],(2,2,3))
  if kind=='wall': return np.concatenate([x.reshape(2,2,2),np.zeros((2,2,1))],axis=2)
  return x.reshape(2,2,3)
 x0={'shared':[1.,0.], 'axis':[1.,0.]*2, 'wall':[1.,0.]*4,'wrap':[1.,0.,0.]*4}[kind]
 def resid(x):
  k=expand(x)
  try:E=eval(k)
  except RuntimeError:return np.full(train.sum(),1e5)
  return E[train]
 lb=np.tile([.001,-1.] if kind!='wrap' else [.001,-1.,0.],len(x0)//(2 if kind!='wrap' else 3)); ub=np.tile([5.,5.] if kind!='wrap' else [5.,5.,5.],len(x0)//(2 if kind!='wrap' else 3));
 r=least_squares(resid,np.maximum(x0,lb+1e-5),bounds=(lb,ub),max_nfev=120,gtol=1e-13,ftol=1e-12,xtol=1e-12)
 e=eval(expand(r.x)); print(name,kind,'params',r.x.round(6),'RMS each pulse',np.sqrt(np.mean(e**2,axis=0)).round(8),'held RMS',np.sqrt(np.mean(e[~train]**2)) if (~train).any() else None,'peak',abs(e).max(),flush=True)
for kind in ['shared','axis','wall','wrap']: fit('all',kind)
for kind in ['wall','wrap']:
 for hold in range(3):fit('hold '+str(hold),kind,hold)
 print('bank labels',np.unique(s['row_bank_id'],return_counts=True))
 for b in np.unique(s['row_bank_id']):fit('bank '+str(b),kind,s['row_bank_id']==b)
