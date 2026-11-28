import pickle,numpy as np
from scipy.linalg import svdvals
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; X=np.c_[np.ones(len(u)),u]; print(k,'u range',u.min(0),u.max(0));
 for key,v in p.items():
  if key=='u':continue
  w=np.ones(len(u));
  for _ in range(15):
   c=np.linalg.lstsq(X*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(v-X@c,axis=1); w=np.minimum(1,1.5*np.median(r)/np.maximum(r,1e-12))**.75
  print(key,'coef',c,'residual',np.sort(r));print('first uv',np.c_[u,v][:4]);
t=s['eval_times']; d=s['train']; z=d['traj']; x=d['x']; y=d['y']; print('t',t,'svd',svdvals(z),'diff',np.diff(t));print('correlations x,z,y',np.corrcoef(np.c_[x,z,y].T)[-1]); print('sorted by y:');print(np.c_[x,z,y][np.argsort(y)][::3]);
for B in [np.c_[np.ones(len(y)),x],np.c_[np.ones(len(y)),z],np.c_[np.ones(len(y)),x,z]]:
 for target in [y,z]:
  pred=B[52:]@np.linalg.lstsq(B[:52],target[:52],rcond=None)[0]; print('LS',B.shape,target.shape,np.mean(abs(pred-target[52:]),axis=0))
