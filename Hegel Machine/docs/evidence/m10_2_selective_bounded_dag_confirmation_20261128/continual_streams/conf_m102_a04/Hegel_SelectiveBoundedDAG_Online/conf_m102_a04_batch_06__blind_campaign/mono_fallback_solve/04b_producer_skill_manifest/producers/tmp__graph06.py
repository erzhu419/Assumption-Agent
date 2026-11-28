import numpy as np,pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def one(x):return np.column_stack((np.ones(len(x)),x))
def poly(x,d):return np.column_stack([np.ones(len(x))]+[x[:,i].prod(1) for j in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),j)])
p=s['panel_a']; U,V=p['u'],p['v']; keep=np.linalg.norm(V,axis=1)<6; u,v=U[keep],V[keep]; print('inliers',len(u));
for j in range(1,5):
 F=poly(u,j); c=ls(F,v); print('A polynomial',j,abs(F@c-v).mean());
 if j==2: print(np.round(c,4))
for name,fn in [('quad',lambda z:poly(z,2)),('cube',lambda z:np.c_[poly(z,2),z**3]),('sepcube',lambda z:np.c_[one(z),z*z,z**3])]:
 f=fn(u); c=ls(f,v); print('A holdout',name,abs(f[-10:]@ls(f[:-10],v[:-10])-v[-10:]).mean())
p=s['panel_b']; b=ls(one(p['u']),p['v']); p=s['panel_c']; c=ls(poly(p['u'],2),p['v1']);
def D(u,q):return np.c_[one(u),u*u,u*u/(1+q*u*u)]
for q0 in ([.25],np.full(4,.25)):
 fit=least_squares(lambda q:(D(p['u'],q)@ls(D(p['u'],q),p['v2'])-p['v2']).ravel(),q0,bounds=(.03,1),gtol=1e-13); print('D',fit.x,abs(fit.fun).mean());
q=fit.x; d=ls(D(p['u'],q),p['v2']); tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']];
for nm,F in [('linear',one),('quad',lambda z:poly(z,2)),('cube',lambda z:np.c_[poly(z,2),z**3])]:
 a=ls(F(u),v); z=np.array([least_squares(lambda zz:(F(zz[None])@a-xx).ravel(),xx,gtol=1e-12).x for xx in x]); w=one(z)@b; print('COORD',nm,'z ranges',np.round([z.min(0),z.max(0)],3));
 for key,h in [('z',z),('w',w),('raw',x)]:
  c1,c2=poly(h,2)@c,D(h,q)@d
  for k,g in [('h',h),('c1',c1),('c2',c2),('c12',np.c_[c1,c2]),('hc12',np.c_[h,c1,c2]),('c12w',np.c_[c1,c2,w])]:
   stats=[]
   for deg in [1,2]:
    A=poly(g,deg); A/=np.maximum(A[:52].std(0),.01); A[:,0]=1
    for reg in [0.001,.05]:
     B=np.linalg.solve(A[:52].T@A[:52]+reg**2*np.eye(A.shape[1]),A[:52].T@Y[:52]); E=abs(A@B-Y); stats.append((deg,reg,*np.round([E[:52,0].mean(),E[:52,1:].mean(),E[52:,0].mean(),E[52:,1:].mean()],4)))
   if key!='raw' or nm=='linear': print(key,k,*stats)
 if nm=='quad': np.savez('/tmp/graph06.npz',x=x,z=z,w=w)
