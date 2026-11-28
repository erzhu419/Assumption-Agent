import numpy as np,pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; yy=np.c_[s['train']['y'],s['train']['traj']]; n=len(x)
def b(x,d):return np.column_stack([np.ones(len(x))]+[x[:,ix].prod(1) for j in range(1,d+1) for ix in combinations_with_replacement(range(x.shape[1]),j)])
def cv(z,d,reg=0):
 a=b(z,d); es=[]
 for ti in np.array_split(np.random.default_rng(17).permutation(n),6):
  tr=np.setdiff1d(np.arange(n),ti); A=a[tr]; w=np.linalg.lstsq(A,yy[tr],rcond=None)[0] if not reg else np.linalg.solve(A.T@A+reg*np.eye(A.shape[1]),A.T@yy[tr]); e=abs(a[ti]@w-yy[ti]);es.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.round(np.mean(es,0),4),np.round(np.max(es,0),4)
for label,z in [('raw',x),('inverse22',(np.sqrt(np.maximum(0,1+.88*x))-1)/.44),('inverse44',(np.sqrt(np.maximum(0,1+1.76*x))-1)/.88),('atanh',np.arctanh(np.clip(x/1.45,-.995,.995)))]:
 for d in (1,2,3,4):print(label,d,cv(z,d,.01 if d>2 else 0))
# Test every single-input univariate nonlinear panel transform using output collinearity with the affine term.
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; U=np.c_[np.ones(len(u)),u];
 for name,v in p.items():
  if name=='u':continue
  W=np.linalg.lstsq(U,v,rcond=None)[0]; K=np.linalg.lstsq(b(u,2),v,rcond=None)[0]; print(k,name,'diag ratios',np.sum(K[[5,9,12,14]]*K[1:5],1)/np.sum(K[1:5]**2,1));print('affine residual largest rows',np.sort(np.linalg.norm(v-U@W,axis=1))[-9:]);
