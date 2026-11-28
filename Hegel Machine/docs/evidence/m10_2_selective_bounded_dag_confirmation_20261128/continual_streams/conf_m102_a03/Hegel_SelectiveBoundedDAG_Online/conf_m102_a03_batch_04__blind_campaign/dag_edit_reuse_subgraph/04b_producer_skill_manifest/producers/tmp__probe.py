import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True)
ls=lambda X,Y:np.linalg.lstsq(X,Y,rcond=None)[0]
def poly(x,d=1):
 return np.column_stack([np.prod(x[:,p],axis=1) for k in range(d+1) for p in combinations_with_replacement(range(x.shape[1]),k)])
def fit(X,Y):
 c=ls(X,Y);r=np.sqrt(np.mean((X@c-Y)**2,axis=1)) if Y.ndim==2 else abs(X@c-Y);return c,np.quantile(r,[0,.25,.5,.75,.9,1])
for p in ('panel_a','panel_b','panel_c'):
 d=s[p];x=d['u']; print(p)
 for key,y in d.items():
  if key=='u':continue
  print(key,'cor',np.corrcoef(x.T,y.T)[:4,4:]);
  for deg in (1,2):
   c,r=fit(poly(x,deg),y);print(deg,r);print(c if deg==1 else '')
  X=poly(x); c=ls(X,y)
  for _ in range(30):
   r=np.linalg.norm(X@c-y,axis=1);w=np.sqrt(np.minimum(1.,max(np.median(r),1.e-12)/np.maximum(r,1.e-12)));c=ls(X*w[:,None],y*w[:,None]);
  print('robust',c,'residual',np.sort(np.linalg.norm(X@c-y,axis=1)))
x=s['train']['x'];y=s['train']['y'];t=s['train']['traj']
for deg in (1,2,3,4):
 X=poly(x,deg);c=ls(X[:52],np.column_stack([y,t])[:52]);r=X@c-np.column_stack([y,t]);print('poly',deg,'train',np.mean(abs(r[:52]),axis=0),'test',np.mean(abs(r[52:]),axis=0));print('y coefs',c[:,0] if deg<=2 else '')
print('traj sv',np.linalg.svd(t,compute_uv=False));print('traj coef',ls(poly(x),t))
