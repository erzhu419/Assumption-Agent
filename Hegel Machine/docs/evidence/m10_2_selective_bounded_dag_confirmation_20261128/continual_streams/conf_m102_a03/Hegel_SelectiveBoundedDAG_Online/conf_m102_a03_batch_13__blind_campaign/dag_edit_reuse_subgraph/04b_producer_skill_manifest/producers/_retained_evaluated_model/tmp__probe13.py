import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; n=len(y)
def fit(F,Y):return np.linalg.lstsq(F,Y,rcond=1e-10)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.c_[np.ones(len(x)),*[np.prod(x[:,p],axis=1) for k in range(1,d+1) for p in itertools.combinations_with_replacement(range(x.shape[1]),k)]]
for p in ('panel_a','panel_b','panel_c'):
 u=s[p]['u']; print(p)
 for k,v in s[p].items():
  if k=='u':continue
  print(k,'range',np.min(v,0),np.max(v,0))
  for d in (1,2,3):
   F=poly(u,d); R=F@fit(F,v)-v; print(d,np.percentile(np.abs(R),[25,50,90,100]))
  if p=='panel_a':
   for j in range(4):
    a=u[:,j]; b=np.roll(u,-1,axis=1)[:,j]; F=np.c_[np.ones(len(a)),a,b,a*b,a**3]; o=least_squares(lambda c:F@c-v[:,j],fit(F,v[:,j]),loss='soft_l1',f_scale=.003);print(j,o.x,np.percentile(abs(o.fun),[50,85,100]))
print('state centered svd',np.linalg.svd(T-T.mean(0),compute_uv=False));print('trajectories minmax',T.min(0),T.max(0))
rng=np.random.default_rng(91); ids=rng.permutation(n); ii=ids[:54]; jj=ids[54:]
for d in (1,2,3,4):
 F=poly(x,d); scale=np.sqrt(np.mean(F[ii]**2,0)); F=F/scale; U,S,V=np.linalg.svd(F[ii],full_matrices=False); print('raw polynomial',d,[(a,np.mean(abs(F[jj]@(V.T@((S/(S*S+a))[:,None]*(U.T@np.c_[y,T][ii])))-np.c_[y,T][jj]),axis=0).round(4).tolist()) for a in (.0001,.01,.1,1.)])
