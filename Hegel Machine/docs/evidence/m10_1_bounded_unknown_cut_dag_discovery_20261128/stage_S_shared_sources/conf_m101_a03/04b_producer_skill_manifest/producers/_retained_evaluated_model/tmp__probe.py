import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=s['train']['x']; Y=s['train']['y']; Q=s['train']['traj']; np.set_printoptions(precision=4,suppress=True,linewidth=155)
def ls(x,y): return lstsq(x,y)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
a=s['panel_a']; A=aff(a['u']); v=a['v']; b=ls(A,v)
for _ in range(15):
 i=np.argsort(np.linalg.norm(A@b-v,axis=1))[:42]; b=ls(A[i],v[i])
a0=b[0]; a1=np.diag(b[1:]); B=s['panel_b']; b=ls(aff(B['u']),B['v']); b0=b[0]; b1=b[1:]; print('Bsv',np.linalg.svd(b1,compute_uv=False)); print('x',X.mean(0),X.std(0))
D={};
for k,x in [('x',X),('cal',(X-a0)/a1),('fcal',X*a1+a0)]:
 D[k]=x; D[k+'B']=x@b1+b0; D[k+'iB']=(x-b0)@np.linalg.inv(b1)
def poly(x,d):
 ids=[p for order in range(1,d+1) for p in combinations_with_replacement(range(4),order)]; return aff(np.array([x[:,p].prod(1) for p in ids]).T)
def cv(z,y,pen=0):
 out=[]
 for it,iv in [(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),iv),iv) for iv in np.array_split(np.random.default_rng(42).permutation(72),4)]:
  zz=z[it]; bb=ls(zz,y[it]) if not pen else np.linalg.solve(zz.T@zz+pen*np.diag(np.r_[0.,np.ones(z.shape[1]-1)]),zz.T@y[it]); out.append(np.mean(abs(z[iv]@bb-y[iv])))
 return np.array(out)
for d in [1,2,3,4]:
 z=poly(X,d); print('poly',d,[(r,cv(z,Y,r),cv(z,Q,r),cv(z,Q[:,-1],r)) for r in [0,.001,.01,.1,1] if d>=3 or r==0])
for k,x in D.items():
 f=np.c_[x,np.sin(x),np.tanh(x),np.sin(2*x),np.tanh(2*x),x*x,x*x*x,np.abs(x)]; z=np.c_[poly(x,2),f[:,4:]]; print(k,'y',cv(z,Y,.005),'Q',cv(z,Q,.005))
for k,p in [('c1',s['panel_c']['v1']),('c2',s['panel_c']['v2'])]:
 x=s['panel_c']['u']; z=poly(x,1); coef=ls(z,p); print(k,'resid SVD',np.linalg.svd(p-z@coef,compute_uv=False),'raw corr',np.corrcoef(np.c_[x,p].T)[:4,4:]);
