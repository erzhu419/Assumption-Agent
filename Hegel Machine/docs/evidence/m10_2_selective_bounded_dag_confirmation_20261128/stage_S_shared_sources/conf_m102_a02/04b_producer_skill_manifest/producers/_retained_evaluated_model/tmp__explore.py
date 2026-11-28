import numpy as np,pickle
from scipy.linalg import lstsq
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; x=s['train']['x']; y=s['train']['y']; h=s['train']['traj']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def feat(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
print('t',t,'svd',np.linalg.svd(h,compute_uv=False)); print('y h corr',np.corrcoef(np.c_[y,h].T)[0]); P={};
for name in ['a','b','c']:
 p=s['panel_'+name]; u=p['u']; print(name,'u',u[:2],u.min(),u.max());
 for k,v in p.items():
  if k=='u':continue
  a=feat(u,1); w=lstsq(a,v)[0]; w=least_squares(lambda w:(a@w.reshape(5,4)-v).ravel(),w.ravel(),loss='soft_l1',f_scale=.01).x.reshape(5,4); P[name+k]=w
  e=v-a@w; print(k,'w',w,'res q',np.quantile(abs(e),[.25,.5,.75,.9,.95,1]),'vsd',v.std(0));print('row err',np.round(np.linalg.norm(e,axis=1),3));
  for d in [1,2,3]:
   a=feat(u,d); w2=lstsq(a[:-6],v[:-6])[0]; print('d',d,'test',np.mean(abs(a[-6:]@w2-v[-6:])),'train',np.mean(abs(a[:-6]@w2-v[:-6])))
for a,n in [(x,'raw')]+[(feat(x,1)@w,k) for k,w in P.items()]:
 for d in [1,2,3]:
  A=feat(a,d); w=lstsq(A[:52],np.c_[y,h][:52])[0]; e=A@w-np.c_[y,h]; print(n,d,'train',np.mean(abs(e[:52]),axis=0),'test',np.mean(abs(e[52:]),axis=0));
