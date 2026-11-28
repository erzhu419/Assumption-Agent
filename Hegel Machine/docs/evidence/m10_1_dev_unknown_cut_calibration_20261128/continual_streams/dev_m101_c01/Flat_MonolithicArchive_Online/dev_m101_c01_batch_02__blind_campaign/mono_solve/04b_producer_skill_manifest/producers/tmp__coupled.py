from operators import *
from scipy.linalg import solve
from scipy.spatial.distance import cdist
import sys
sys.path.insert(0,'/holdout_task')
import batch_solver as bs
d=S['train']; x=d['x']; y=d['y']; z=d['traj']; t=S['eval_times']; n=len(x)
print('times',t,'telemetry',S['telemetry_vec'],'summary',d['summary'])
rng=np.random.default_rng(773); sets=[np.arange(56,72)]+[rng.permutation(n)[:16] for _ in range(8)]+[np.argsort(abs(x[:,j]))[-16:] for j in range(4)]+[np.argsort(np.linalg.norm(x-x[np.argmax(abs(x[:,j]))],axis=1))[:16] for j in range(4)]
f=bs.quadratic(x); P=f@f.T; D=cdist(x,x,'sqeuclidean'); dr,iv=bs.panel_coordinates(S,x); K=P+np.exp(-D/(2*1.6**2))+.2*bs.additive_kernel(iv,1.5); L=P+.03*bs.additive_kernel(dr,.45); K0=L+.1*np.exp(-D/(2*1.1**2)); K1=L+np.exp(-D/(2*1.6**2)); values={}
for te in sets:
 tr=np.setdiff1d(np.arange(n),te); b=bs.temporal_basis(t,z[tr]); a=ls(b,z[tr].T).T
 fn=lambda g,e:g[np.ix_(te,tr)]@solve(g[np.ix_(tr,tr)]+e*np.eye(len(tr)),np.eye(len(tr)),assume_a='pos')
 ka=fn(K,.002); ky=.5*(fn(K0,.0003)+fn(K1,.0003)); ap=ka@a
 def record(k,p):values.setdefault(k,[]).append(np.mean(abs(p-y[te])))
 record('baseline',ky@y[tr])
 for k in ('a','x+a','x*q+a','x*q+a+cross','a+x+quot','a*x+quot'):
  def feat(v,m):
   if k=='a':return bs.quadratic(m)
   if k=='x+a':return np.c_[affine(v),bs.quadratic(m)[:,1:]]
   if k=='x*q+a':return np.c_[bs.quadratic(v),bs.quadratic(m)[:,1:]]
   if k=='x*q+a+cross':return np.c_[bs.quadratic(v),bs.quadratic(m)[:,1:],(v[:,:,None]*m[:,None,:]).reshape(len(v),-1)]
   if k=='a+x+quot':return np.c_[affine(v),bs.quadratic(m)[:,1:],affine(v)/(1+m[:,0,None]**2),affine(v)/(1+m[:,1,None]**2)]
   return np.c_[affine(v),bs.quadratic(m)[:,1:],(v[:,:,None]*m[:,None,:]).reshape(len(v),-1),affine(v)/(1+m[:,0,None]**2),affine(v)/(1+m[:,1,None]**2)]
  F=feat(x[tr],a); Ft=feat(x[te],ap)
  for l in (.001,.03,.2):
   coef=solve(F.T@F+l*np.eye(F.shape[1]),F.T@y[tr],assume_a='pos'); e=y[tr]-F@coef
   record((k,l,'direct'),Ft@coef)
   record((k,l,'residual'),Ft@coef+ky@e)
   if te[0]==56 and l==.001:print(k,'scalar train rms',np.sqrt(np.mean(e*e)))
for k,v in sorted(values.items(),key=lambda q:np.mean(q[1][:9])+.25*np.mean(q[1][9:]))[:18]:print(k,np.round([v[0],np.mean(v[1:9]),max(v[1:9]),np.mean(v[9:]),max(v[9:])],6))
