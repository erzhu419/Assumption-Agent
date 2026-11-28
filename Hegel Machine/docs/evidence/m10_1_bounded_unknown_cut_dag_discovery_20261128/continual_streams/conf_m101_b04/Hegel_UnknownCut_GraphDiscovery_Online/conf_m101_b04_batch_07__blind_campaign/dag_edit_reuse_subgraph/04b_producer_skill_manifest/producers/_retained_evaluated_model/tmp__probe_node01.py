import sys,pickle,itertools
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=run_node({'__input__':s})['x_tr']; u=s['panel_b']['u']; v=s['panel_b']['v']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; np.set_printoptions(precision=5,suppress=True,linewidth=140)
def fit(a,b):
 c=np.linalg.lstsq(a,b,rcond=None)[0]; e=b-a@c; return c,np.mean(abs(e),axis=0)
def poly(a,d): return np.column_stack([np.prod(a[:,q],axis=1) for k in range(d+1) for q in itertools.combinations_with_replacement(range(a.shape[1]),k)])
print('ranges',x.min(0),x.max(0), 'panel ranges',u.min(0),u.max(0)); print('panel rows',np.c_[u,v][:10]); print('panel errors', [fit(poly(u,d),v)[1] for d in (1,2)])
for k in (1,2,3):
 a=poly(x,k); c,e=fit(a,np.c_[y,z]); h=np.sum(np.linalg.qr(a)[0]**2,axis=1); print('raw polynomial',k,'in MAE',e,'LOO',np.mean(abs((np.c_[y,z]-a@c)/(1-h[:,None])),axis=0))
print('trajectory singular values',np.linalg.svd(z,compute_uv=False)); print('y from traj',fit(np.c_[np.ones(len(x)),z],y)[1]); print('raw first trajectories',np.c_[x,y,z][:10]); print('second differences',np.diff(z,n=2)[:8])
for k in range(1,5):
 a=np.column_stack([z[:,i+1:8-k+i+1].ravel() for i in range(k)]); print('trajectory recurrence order',k,fit(a,z[:,k:].ravel()))
