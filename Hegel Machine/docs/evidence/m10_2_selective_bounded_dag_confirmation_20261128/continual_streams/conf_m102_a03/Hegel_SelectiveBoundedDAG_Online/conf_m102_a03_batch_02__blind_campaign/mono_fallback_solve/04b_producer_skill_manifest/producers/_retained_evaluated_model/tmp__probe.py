import pickle,numpy as np,sys
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive')
from conf_m102_a03_batch_01_solver import _calibration
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,t=[tr[k] for k in ('x','y','traj')]; n=52
np.set_printoptions(precision=5,suppress=True,linewidth=170)
def A(x): return np.c_[np.ones(len(x)),x]
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def P(x,d): return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
a=_calibration(s['panel_a']); p=s['panel_b']; b=fit(A(p['u']),p['v']); z=np.linalg.solve(a[1:].T,(x-a[0]).T).T; w=A(z)@b
print('times',s['eval_times'],'calib',a,'calib res',np.sort(np.linalg.norm(A(s['panel_a']['u'])@a-s['panel_a']['v'],axis=1))); print('B ortho',b[1:].T@b[1:]); print('singvals traj',np.linalg.svd(t,compute_uv=False))
for j in ('v1','v2'):
 p=s['panel_c']; u,v=p['u'],p[j]; print('C',j,'range',v.min(0),v.max(0)); print(np.c_[u,v][:12]);
 for k in (1,2,3):
  f=P(u,k); M=np.linalg.pinv(f); err=v-f@M@v; lev=np.sum(f*M.T,axis=1); print(k,'fit/loo',np.std(err),np.mean(np.abs(err/np.maximum(1e-12,1-lev[:,None]))));
for name,q in [('x',x),('z',z),('w',w)]:
 for d in [1,2,3,4]:
  f=P(q,d); c=fit(f[:n],y[:n]); r=f@c-y; h=fit(f[:n],t[:n]); print(name,d,'train/hold',np.mean(abs(r[:n])),np.mean(abs(r[n:])), 'traj',np.mean(abs((f@h-t)[n:])))
  if d==2: print('quadratic coeff',c)
print('y sorted',np.sort(y)); print('traj linear coefs in w',fit(A(w),t)); print('summary discrep',tr['summary']-np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()]))
