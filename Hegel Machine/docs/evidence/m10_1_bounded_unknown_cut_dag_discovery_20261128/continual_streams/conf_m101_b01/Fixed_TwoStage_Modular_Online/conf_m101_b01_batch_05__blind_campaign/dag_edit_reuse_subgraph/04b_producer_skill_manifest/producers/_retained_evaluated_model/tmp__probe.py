import pickle, numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def poly(x,d): return np.column_stack([np.prod(x[:,inds],axis=1) for n in range(d+1) for inds in combinations_with_replacement(range(x.shape[1]),n)])
def err(X,Y,inds):
 z=np.array([j for j in range(len(Y)) if j not in inds]); c=np.linalg.lstsq(X[z],Y[z],rcond=None)[0]; return np.mean(abs((X@c-Y)[inds]),axis=0)
for name in ('panel_a','panel_b','panel_c'):
 p=s[name]; print(name, 'ranges', np.min(p['u'],axis=0),np.max(p['u'],axis=0))
 for k in p.keys()-{'u'}:
  x,y=p['u'],p[k]; print(k,'center',y.mean(0),'std',y.std(0));
  for d in (1,2):
   X=poly(x,d); c=np.linalg.lstsq(X,y,rcond=None)[0]; r=X@c-y; h=(X*np.linalg.pinv(X).T).sum(1); print(d,'LOO',np.mean(abs(r/(1-h[:,None])),0),'row errors',np.sort(np.linalg.norm(r,axis=1))[-8:]); print('coefs',c if d==1 else c[5:])
tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']; Y=np.column_stack([y,t]); print('telemetry',s['telemetry_vec'],'times',s['eval_times'],'summary',tr['summary']); print('trajectory singular values',np.linalg.svd(t,compute_uv=False)); print('first trajs',t[:8]); print('train/test ranges',x.min(0),x.max(0),s['test']['x'].min(0),s['test']['x'].max(0));
for d in (1,2,3,4):
 X=poly(x,d); print('degree',d,'suffix',err(X,Y,list(range(54,72))), '6fold',np.mean([err(X,Y,list(range(j,72,6))) for j in range(6)],axis=0));
