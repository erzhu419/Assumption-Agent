import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('keys',s.keys()); print('times',s['eval_times']); print('telemetry',s['telemetry_vec'])
def poly(x,d):
 k=[t for m in range(d+1) for t in cwr(range(x.shape[1]),m)];return np.column_stack([np.prod(x[:,t],axis=1) for t in k]),k
def ls(X,y):return np.linalg.lstsq(X,y,rcond=None)[0]
u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.max(abs(v),axis=1)<3;u,v=u[keep],v[keep]; X,k=poly(u,3);b=ls(X,v); print('A cubic',[(a,np.round(z,5)) for a,z in zip(k,b) if max(abs(z))>.003]); print('A noise',np.quantile(abs(X@b-v),[.5,.9,1]));
u2,v2=[s['panel_b'][k] for k in ('u','v')]; d=lambda x,w:np.column_stack((np.ones(len(x)),np.cos(x*w))); f=lambda w:(d(u2,w)@ls(d(u2,w),v2)-v2).ravel(); w=least_squares(f,[2.5]).x; B=ls(d(u2,w),v2); print('B',w,B,'sing values',np.linalg.svd(B[1:],compute_uv=False))
def cal(x):
 zz=np.linalg.solve(b[1:5].T,(x-b[0]).T).T
 for i in range(len(x)):
  def err(z):return poly(z[None,:],3)[0][0]@b-x[i]
  zz[i]=least_squares(err,zz[i],gtol=1e-12).x
 return zz
z=cal(s['train']['x']); c=d(z,w)@B; print('ranges',z.min(0),z.max(0),c.min(0),c.max(0));print('z x max',np.max(abs(z-s['train']['x']),0));
y=s['train']['y']; T=s['train']['traj']; i=np.arange(len(y)); tr=i[:52]; te=i[52:]
def score(name,F,report=False):
 a=ls(F[tr],y[tr]); p=F@a; e=F@ls(F[tr],T[tr])-T; print(name,F.shape[1],np.mean(abs(p[tr]-y[tr])),np.mean(abs(p[te]-y[te])),np.mean(abs(e[te])),np.round(a,3) if report else '')
for name,q in [('c',c),('cos',np.cos(z*w)),('z',z),('x',s['train']['x'])]:
 for degree in range(1,4):score((name,degree),poly(q,degree)[0],name=='c' and degree<=2)
 a=poly(q,1)[0]; lin=q[:,:2]*q[:,2:]; dot=lin.sum(1); score((name,'dot'),np.column_stack((a,dot)),True)
 for f in [np.sin,np.cos,np.tanh,np.arctan,lambda x:x/(1+x*x),lambda x:x**3,lambda x:abs(x),lambda x:x**2]:score((name,f.__name__),np.column_stack((a,dot,f(dot),f(q) if f in [np.sin,np.cos,np.tanh,np.arctan] else lin)))
np.savez('/tmp/readprobe.npz',z=z,c=c,y=y,T=T)
