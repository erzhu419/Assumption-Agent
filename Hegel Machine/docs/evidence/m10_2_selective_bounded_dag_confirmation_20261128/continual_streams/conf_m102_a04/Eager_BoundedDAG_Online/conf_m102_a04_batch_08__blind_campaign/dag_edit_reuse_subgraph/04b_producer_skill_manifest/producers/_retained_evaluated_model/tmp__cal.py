import numpy as np,pickle
from itertools import combinations,combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=s['panel_a']; u,v=a['u'],a['v']; print('a shape',u.shape,v.shape); print(np.round(np.c_[u,v],3)); print('keys',s.keys()); print('telemetry',s['telemetry_vec']); print('panel sizes',[(k,{j:np.shape(x) for j,x in s[k].items()}) for k in ['panel_a','panel_b','panel_c']]);
def one(x):return np.c_[np.ones(len(x)),x]
def lin(f,y):return np.linalg.lstsq(f,y,rcond=1e-10)[0]
def pairs(x):return np.stack([x[:,i]*x[:,j] for i,j in combinations(range(x.shape[1]),2)],1)
keep=np.linalg.norm(v,axis=1)<5
print('keep',keep.sum()); u=u[keep]; v=v[keep]
for nm,x,y in [('forward',u,v),('inverse',v,u)]:
 def test(name,fn):
  F=fn(x); h=np.sum(np.linalg.svd(F,full_matrices=False)[0][:,:np.linalg.matrix_rank(F)]**2,axis=1); e=F@lin(F,y)-y; print(nm,name,np.sqrt(np.mean(e**2,axis=0)),np.mean(abs(e/np.maximum(1-h[:,None],.001))))
 for n,func in [('linear',one),('diag2',lambda z:np.c_[one(z),z*z]),('diag3',lambda z:np.c_[one(z),z*z,z**3]),('diag5',lambda z:np.c_[one(z),z*z,z**3,z**4,z**5]),('full2',lambda z:np.c_[one(z),z*z,pairs(z)]),('diag3pair',lambda z:np.c_[one(z),z*z,z**3,pairs(z)]),('signed2',lambda z:np.c_[one(z),z*abs(z)]),('sin',lambda z:np.c_[one(z),np.sin(2*z)]),('saturate',lambda z:np.c_[one(z),z/(1+z*z),z*z/(1+z*z)]),('full3',lambda z:np.c_[one(z),z*z,pairs(z),*[np.prod(z[:,ix],axis=1) for ix in combinations_with_replacement(range(4),3)]])]:test(n,func)
 print('affine coeff',lin(one(x),y)); print('diag5 coeff',lin(np.c_[one(x),x*x,x**3,x**4,x**5],y))
