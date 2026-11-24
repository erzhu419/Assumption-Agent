import numpy as np
from scipy.optimize import minimize
z=np.load('/holdout_task/public_graph.npz'); X=z['X']; A=z['gt_adj']; S0=z['gt_S']; n,p=X.shape
C=np.cov(X,rowvar=False); P=np.linalg.inv(C)
def obj(v):
 U=v.reshape(p,2); H=P+U@U.T
 return np.sum(np.abs(H[np.triu_indices(p,1)])) + .001*np.sum(U*U)
for seed in range(3):
 res=minimize(obj,np.random.default_rng(seed).normal(0,.3,(p,2)).ravel(),method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-12})
 U=res.x.reshape(p,2); H=P+U@U.T
 vals=np.abs(H[np.triu_indices(p,1)]); print(res.fun,res.success,np.quantile(vals,[.5,.8,.9,.95,.98]))
 for t in [.08,.12,.16,.2,.24,.28,.32]:
  B=np.abs(H)>t; np.fill_diagonal(B,False); tp=np.sum(B&A)//2; fp=np.sum(B&~A)//2; fn=np.sum(A&~B)//2; f=2*tp/(2*tp+fp+fn); print(t,round(f,3),tp,fp,fn)
