import itertools as it, pickle, runpy
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['c']; y=s['train']['y']; n=len(y)
a,b,d,e=c.T; p=a*d+b*e; base=np.c_[np.ones(n),c,p]; tr=np.arange(52); va=np.arange(52,n)
ix=list(it.combinations_with_replacement(range(4),3)); D=np.column_stack([np.prod(c[:,i],axis=1) for i in ix]); names=[''.join(map(str,i)) for i in ix]; rng=np.random.default_rng(724)
def evaluate(X):
 beta=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0]; err=X@beta-y
 return np.sqrt(np.mean(err[tr]**2)),np.mean(abs(err[va])),beta
for B in (base,np.c_[base,a*d-b*e,a*e,b*d]):
 print('base',B.shape[1]); print('all cubic',evaluate(np.c_[B,D])); Q=np.linalg.qr(B[tr])[0]; E=D[tr]-Q@(Q.T@D[tr]); r=y[tr]-Q@(Q.T@y[tr]);
 for k in range(1,6):
  best=[]
  for j in it.combinations(range(len(ix)),k):
   X=E[:,j]; z=np.linalg.lstsq(X,r,rcond=None)[0]; best.append((np.linalg.norm(X@z-r),j))
  for _,j in sorted(best)[:3]:
   X=np.c_[B,D[:,j]]; rm,ma,beta=evaluate(X); print(k,[names[i] for i in j],np.round(beta,5),round(rm,6),round(ma,6))
print('cubic coefficients with fixed paired quadratic',list(zip(names,evaluate(np.c_[base,D])[2][6:].round(5))))
print('nested support stability (training-only BIC):')
for v in np.array_split(rng.permutation(tr),5):
 t=np.setdiff1d(tr,v); best=(np.inf,None,None)
 for k in range(5):
  for j in it.combinations(range(len(ix)),k):
   X=np.c_[base,D[:,j]]; z=np.linalg.lstsq(X[t],y[t],rcond=None)[0]; mse=np.mean((X[t]@z-y[t])**2); score=len(t)*np.log(max(mse,1e-20))+k*np.log(len(t));
   if score<best[0]:best=(score,j,np.mean(abs(X[v]@z-y[v])))
 print([names[i] for i in best[1]],best[2])
