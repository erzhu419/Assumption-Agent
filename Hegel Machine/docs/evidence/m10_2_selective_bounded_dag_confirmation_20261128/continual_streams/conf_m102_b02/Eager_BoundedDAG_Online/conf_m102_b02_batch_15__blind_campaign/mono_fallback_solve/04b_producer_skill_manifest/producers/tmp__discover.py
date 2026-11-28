import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=160)
def d(*x):return np.column_stack((np.ones(len(x[0])),*x))
def fit(x,y):return np.linalg.lstsq(x,y,rcond=1e-11)[0]
def af(u,q):return d(u,u**3,np.sin(q[0]*u),np.tanh(q[1]*u))
def bf(u,q):return d(u*u,np.cos(q[0]*u),np.exp(-q[1]*u*u))
def cf(u,q):return d(u,u*u,np.cos(q[0]*u),np.exp(-q[1]*u*u))
def c1f(u):return d(u*np.roll(u,-1,axis=1))
def op(key,vkey,F,q0):
 u,v=s[key]['u'],s[key][vkey]; keep=np.arange(len(u)); f=F(u,q0)
 if key=='panel_a':
  for _ in range(8):
   err=np.linalg.norm(f@fit(f[keep],v[keep])-v,axis=1);keep=np.argsort(err)[:40]
 def res(q):
  f=F(u[keep],q);return (f@fit(f,v[keep])-v[keep]).ravel()
 for _ in range(3):
  q=least_squares(res,q0,bounds=([.3,.3],[5.,6.]),gtol=1e-12).x; f=F(u,q); w=fit(f[keep],v[keep]); err=np.linalg.norm(f@w-v,axis=1)
  if key=='panel_a': keep=np.flatnonzero(err<max(.001,4*np.median(err)))
 print(key,vkey,'q',q,'res',np.quantile(err,[0,.5,.8,.9,1]),'n',len(keep),'coef',w)
 return lambda x:F(x,q)@w,w
A,aw=op('panel_a','v',af,[2.15,2.15]); B,bw=op('panel_b','v',bf,[2.15,2.15]); C2,cw=op('panel_c','v2',cf,[2.15,2.15]); C1=lambda u:c1f(u)@fit(c1f(s['panel_c']['u']),s['panel_c']['v1'])
tr=s['train']; x=np.r_[tr['x'],s['test']['x']]; z0=np.linalg.solve(aw[1:5].T,(x-aw[0]).T).T
z=np.array([least_squares(lambda z:A(z[None])[0]-xi,np.clip(zi,-1.3,1.3),bounds=(-2,2),gtol=1e-12).x for xi,zi in zip(x,z0)])
print('z range',z.min(0),z.max(0),'res',np.max(np.abs(A(z)-x)))
b,c,h=B(z),C1(z),C2(z)
t=s['eval_times']; T=tr['traj']; F=lambda p:-np.expm1(-t[:,None]*p); pp=least_squares(lambda p:(F(p)@fit(F(p),T.T)-T.T).ravel(),[.5,2],bounds=([.01,.02],[5,8]),gtol=1e-12).x; f=F(pp); m=fit(f,T.T).T
print('time',pp,'MAE',np.mean(abs(m@f.T-T)))
np.savez('/tmp/design.npz',x=x,z=z,b=b,c=c,h=h,m=m,y=tr['y'],f=f,T=T)
print('z,b,c,h sample',np.c_[z,b,c,h][:5])
for nm,v in list(zip(['x','z','b','c','h'],[x,z,b,c,h]))+[('all',np.c_[z,b,c,h])]:
 v=d(v); o=np.c_[tr['y'],m]; pred=v[52:72]@fit(v[:52],o[:52]); print(nm,'MAE',np.mean(abs(pred-o[52:]),axis=0),np.mean(abs(pred[:,1:]@f.T-T[52:])))
