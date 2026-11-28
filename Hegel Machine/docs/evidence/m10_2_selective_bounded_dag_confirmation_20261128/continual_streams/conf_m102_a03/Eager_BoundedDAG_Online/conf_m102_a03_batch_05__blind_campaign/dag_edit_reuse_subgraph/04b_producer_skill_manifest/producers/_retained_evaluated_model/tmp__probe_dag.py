import numpy as np,pickle,sys,itertools as it
from scipy.linalg import svd
sys.path.insert(0,'/holdout_task/_frozen_nodes'); import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_b']; u,v=p['u'],p['v']; b=n._fit(n._affine(u),v); a=n._calibration(s['panel_a']); x=np.linalg.solve(a[1:].T,(s['train']['x']-a[0]).T).T; y=s['train']['y']; t=s['train']['traj']; U,S,V=np.linalg.svd(t,full_matrices=False); yy=U[:,0]*S[0]
def mat(x,terms):return np.column_stack([np.prod(x[:,i],axis=1) if i else np.ones(len(x)) for i in terms])
def terms(d):return [i for k in range(d+1) for i in it.combinations_with_replacement(range(4),k)]
def scores(f,y,split=None):
 tr=np.arange(len(y)); ii,jj=(tr[:split],tr[split:]) if split else (tr,tr)
 return np.round(np.mean(np.abs(f[jj]@n._fit(f[ii],y[ii])-y[jj]),axis=0),5)
for d in (1,2,3,4):
 f=mat(x,terms(d)); print('study poly',d,scores(f,y),scores(f,y,48),scores(f,yy),scores(f,yy,48));
 g=mat(u,terms(d)); print('panel poly',d,scores(g,v),scores(g,v,18));
 if d in (2,3):
  for name,f0,out in [('panel',g,v),('scalar',f,y),('amp',f,yy)]:
   c=n._fit(f0,out); print(name,[(terms(d)[i],np.round(c[i],3).tolist()) for i in np.argsort(np.linalg.norm(c.reshape(len(c),-1),axis=1))[-12:][::-1]])
print('time PC',V[:2]); print('Panel C stats');
for k in ('v1','v2'):
 q=s['panel_c']; print(k,scores(mat(q['u'],terms(1)),q[k]),scores(mat(q['u'],terms(2)),q[k],18));
for name,h in [('sin',np.sin),('tanh',np.tanh),('sinh',np.sinh),('cos',np.cos)]:
 for w in (.5,1,1.25,1.5,2,2.5,3):
  def feat(z): return np.c_[mat(z,terms(2)),h(w*z)]
  f=feat(x); g=feat(u); print(name,w,scores(f,y,48),scores(f,yy,48),scores(g,v,18))
