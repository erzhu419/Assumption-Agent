import numpy as np
from scipy.optimize import least_squares
def d(*x):return np.column_stack((np.ones(len(x[0])),*x))
def fit(x,y):return np.linalg.lstsq(x,y,rcond=1e-11)[0]
def af(u,q):return d(u,u**3,np.sin(q[0]*u),np.tanh(q[1]*u))
def bf(u,q):return d(u*u,np.cos(q[0]*u),np.exp(-q[1]*u*u))
def cf(u,q):return d(u,u*u,np.cos(q[0]*u),np.exp(-q[1]*u*u))
def c1f(u):return d(u*np.roll(u,-1,axis=1))
def op(s,key,vkey,F,q0):
 u,v=s[key]['u'],s[key][vkey]; keep=np.arange(len(u)); f=F(u,q0)
 if key=='panel_a':
  for _ in range(8):
   err=np.linalg.norm(f@fit(f[keep],v[keep])-v,axis=1);keep=np.argsort(err)[:40]
 def res(q):
  f=F(u[keep],q);return (f@fit(f,v[keep])-v[keep]).ravel()
 for _ in range(3):
  q=least_squares(res,q0,bounds=([.3,.3],[5.,6.]),gtol=1e-12).x; f=F(u,q); w=fit(f[keep],v[keep]); err=np.linalg.norm(f@w-v,axis=1)
  if key=='panel_a': keep=np.flatnonzero(err<max(.001,4*np.median(err)))
 return lambda x:F(x,q)@w,w

from itertools import combinations_with_replacement

def coordinates(s,x):
 A,aw=op(s,'panel_a','v',af,[2.15,2.15]); B,_=op(s,'panel_b','v',bf,[2.15,2.15]); C2,_=op(s,'panel_c','v2',cf,[2.15,2.15]); pc=s['panel_c']; cw=fit(c1f(pc['u']),pc['v1']); z0=np.linalg.solve(aw[1:5].T,(x-aw[0]).T).T
 z=np.array([least_squares(lambda z:A(z[None])[0]-xi,np.clip(zi,-1.3,1.3),bounds=(-2,2),gtol=1e-12).x for xi,zi in zip(x,z0)])
 return dict(z=z,b=B(z),c=c1f(z)@cw,h=C2(z))

def quad(x):
 return np.stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2)],axis=1)

def temporal(s):
 t=np.asarray(s['eval_times']); T=np.asarray(s['train']['traj']); F=lambda p:-np.expm1(-t[:,None]*p); p=least_squares(lambda p:(F(p)@fit(F(p),T.T)-T.T).ravel(),[.42,1.81],bounds=([.10,.7],[.7,4.]),gtol=1e-12).x; f=F(p)
 return f,fit(f,T.T).T

def residpred(F,K,y):
 n=len(y); q=np.linalg.pinv(F[:n],rcond=1e-9); v=q@y; R=np.eye(n)-F[:n]@q
 P=K[:, :n]-F@(q@K[:n,:n]); return F@v+P@R@np.linalg.solve(R@K[:n,:n]@R+.1*np.eye(n),R@y)

def solve_batch(study: dict) -> dict:
 tr=study['train']; n=len(tr['x']); cc=coordinates(study,np.r_[tr['x'],study['test']['x']]); z,b,c,h=[cc[k] for k in 'zbch']; f,m=temporal(study); qb=quad(b)
 K=np.zeros((len(z),len(z)))
 for X,w in [(z,.6),(b,.35),(c,.15),(h,.15)]:
  X=X/np.maximum(X[:n].std(0),.1); D=((X[:,None]-X[None,:])**2).sum(-1); K+=w*(np.exp(-D/8)+np.exp(-D/2))
 pred=np.column_stack([residpred(d(z),K,m[:,0]),residpred(d(qb),K,m[:,1])]); y0=d(z[:n],qb[:n],m); y=d(z,qb,pred)[n:]@fit(y0,tr['y']); state=pred[n:]@f.T; end=state[:,-1]
 return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
