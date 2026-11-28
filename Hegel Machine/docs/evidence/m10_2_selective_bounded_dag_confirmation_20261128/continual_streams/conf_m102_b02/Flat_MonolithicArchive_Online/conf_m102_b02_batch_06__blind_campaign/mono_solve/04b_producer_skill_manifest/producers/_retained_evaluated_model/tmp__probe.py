import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def I(x):return np.column_stack((np.ones(len(x)),x))
def ls(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def test_panel(u,v):
 p=np.roll(u,-1,axis=1); X=[('lin',I(u)),('pair',I(u*p)),('cubic',I(np.c_[u,u*p,u**3])),('quad',I(np.c_[u,np.array([u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]).T]))];
 for n,d in X:
  keep=np.ones(len(u),bool)
  for j in range(5):
   e=v-d@ls(d[keep],v[keep]); keep=np.max(abs(e),axis=1)<0.8
   if keep.sum()<d.shape[1]:keep[:]=1;break
  e=v-d@ls(d[keep],v[keep]);print(n,'kept',keep.sum(),'mae',np.abs(e[keep]).mean(),'col',np.median(np.abs(e),axis=0))
 for name,fn in [('sin',np.sin),('asinh',np.arcsinh),('tanh',np.tanh),('sat',lambda a:a/(1+a*a))]:
  for family in ['linear_plus','mixed']:
   def d(q):return I(np.c_[u,fn(u*np.exp(q[0]))]) if family=='linear_plus' else I(u-q[1]*fn(u*np.exp(q[0])))
   def r(q):
    a=d(q);return (a@ls(a,v)-v).ravel()
   fits=[least_squares(r,[np.log(a),.3],bounds=([-3,-3],[2,3]),max_nfev=70) for a in [.7,1.5,3.5]];f=min(fits,key=lambda f:f.cost);print(name,family,np.round([np.exp(f.x[0]),f.x[1],np.mean(abs(f.fun))],6))
np.set_printoptions(precision=4,suppress=True)
for k in ['panel_b','panel_c']:
 for j in s[k]:
  if j=='u':continue
  print(k,j);test_panel(s[k]['u'],s[k][j])
print('calibration');u,v=s['panel_a'].values(); p=np.roll(u,-1,axis=1)
for n,d in [('affine',I(u)),('quad/cubic',I(np.c_[u,u*p,u**3]))]:
 keep=np.ones(len(u),bool)
 for it in range(5):
  b=ls(d[keep],v[keep]);e=np.max(abs(v-d@b),axis=1);keep=e<.75
 b=ls(d[keep],v[keep]); print(n,b,np.sort(np.max(abs(v-d@b),axis=1)))
print('traj singular values',np.linalg.svd(s['train']['traj'],compute_uv=False))
