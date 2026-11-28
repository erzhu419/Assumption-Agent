import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=120)
for name in ('panel_a','panel_b','panel_c'):
 p=s[name]; u=p['u']; D=np.c_[np.ones(len(u)),u]; print(name,'u range',u.min(0),u.max(0))
 for k,v in p.items():
  if k=='u':continue
  w=np.ones(len(u),bool)
  for j in range(4):
   b=np.linalg.lstsq(D[w],v[w],rcond=None)[0]; r=np.linalg.norm(v-D@b,axis=1); w=r<3*np.median(r)+.0001
  print(k,'v range',v.min(0),v.max(0),'coef',b,'sorted affine residual',np.sort(r))
t=s['eval_times']; Y=s['train']['traj'];
def basis(r):return -np.expm1(-t[:,None]*r)
def fit(r):
 Q=basis(r); return (Y.T-Q@np.linalg.lstsq(Q,Y.T,rcond=None)[0]).ravel()
r=least_squares(fit,[.4,1.85],bounds=(.01,10)).x; Q=basis(r); A=np.linalg.lstsq(Q,Y.T,rcond=None)[0].T
print('rates',r,'projected MAE',np.mean(abs(fit(r))),'amplitudes and target corr',np.corrcoef(np.c_[A,s['train']['y']].T));print('amplitudes and target first rows',np.c_[A,s['train']['y']][:8]); print('x ranges',s['train']['x'].min(0),s['train']['x'].max(0),s['test']['x'].min(0),s['test']['x'].max(0))
