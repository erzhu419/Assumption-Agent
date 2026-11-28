import pickle, numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_c']['u']; v=s['panel_c']['v2']; p=list(combinations(range(4),2)); l=x[:,[i for i,j in p]]; r=x[:,[j for i,j in p]]; a=np.c_[np.ones(len(x)),x]; q=x*x; uv=l*r; n=np.sum(q,axis=1,keepdims=True)
res=[]
def score(name,fn):
 def metrics(k,loo=False):
  f=fn(k); Q=np.linalg.svd(f,full_matrices=False)[0]; Q=Q[:,:np.linalg.matrix_rank(f)]; e=v-Q@(Q.T@v)
  if loo:e=e/np.maximum(1-np.sum(Q**2,axis=1),1e-9)[:,None]
  return np.sqrt(np.mean(e**2))
 grid=[0,.015,.06,.22,.6,1.2,2.15,4,8,16,40]; j=np.argmin([metrics(k) for k in grid]); fit=minimize_scalar(metrics,bounds=(max(1e-8,grid[max(j-1,0)]),grid[min(j+1,len(grid)-1)]),method='bounded'); k=fit.x; res.append((metrics(k,True),fit.fun,k,name))
for nam,w in [('pairs',uv),('pairnorm',l*l+r*r),('sum',l+r),('diff',l-r),('sumsq',(l+r)**2),('diffsq',(l-r)**2),('pairdiff',l*l-r*r),('pairs abs',abs(uv))]:
 for dn,d in [('l2',l*l),('r2',r*r),('pnorm',l*l+r*r),('pair2',uv**2),('abspair',abs(uv)),('sum2',(l+r)**2),('diff2',(l-r)**2),('absdiff',abs(l-r)),('abssum',abs(l+r)),('absnorm',abs(l)+abs(r)),('norm',n)]:
  for baseline in [a,np.c_[a,q]]:score((nam,dn,baseline.shape[1]),lambda k:np.c_[baseline,w/(1+k*d)])
for dn,d in [('norm',n),('norm2',n*n),('maxabs',np.max(abs(x),axis=1,keepdims=True)),('l1',np.sum(abs(x),axis=1,keepdims=True)),('sum2',x.sum(1,keepdims=True)**2)]:
 for nm,w in [('lin',a),('quad',np.c_[a,q,uv]),('homo',np.c_[q,uv]),('pairs',uv)]:
  score((nm,dn,'all'),lambda k:w/(1+k*d));score((nm,dn,'nonlinear'),lambda k:np.c_[a,w/(1+k*d)])
for i in range(4):
 for j in range(i+1,4):
  for sign in [-1,1]:
   for c in [.5,1,2]:
    t=x.copy(); t[:,i]=x[:,i]+sign*c*x[:,j];t[:,j]=x[:,i]-sign*c*x[:,j]; score(('rotated square',i,j,sign,c),lambda k:np.c_[a,t*t/(1+k*t*t)])
for row in sorted(res,key=lambda x:x[0])[:28]: print(row)
for row in sorted(res,key=lambda x:x[1])[:12]: print('fit',row)
