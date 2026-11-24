import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from prony_rheology import _memory_design
from scipy.optimize import lsq_linear
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment'];t=e['train_t'];x=e['train_strain'];y=e['train_stress_raw'];tx=e['test_t'];xx=e['test_strain'];truth=p['gt_test_stress']
for n in [10,20,40,80,160,300]:
 taus=np.geomspace(.02,20,n);D=_memory_design(t,x,taus); A=np.c_[x,D,np.ones(len(t)),t]; Dx=_memory_design(tx,xx,taus); Ax=np.c_[xx,Dx]
 for mode in ['ls','huber']:
  w=np.ones(len(y))
  for it in range(20 if mode=='huber' else 1):
   fit=lsq_linear(A*np.sqrt(w[:,None]),y*np.sqrt(w),bounds=(np.r_[np.zeros(n+1),[-np.inf,-np.inf]],np.inf),max_iter=500,tol=1e-12);b=fit.x
   r=y-A@b; med=np.median(r);scale=1.4826*np.median(abs(r-med));u=abs(r-med)/(1.345*scale+1e-12);w=np.minimum(1,1/np.maximum(u,1e-12))
  pred=Ax@b[:n+1];err=np.sqrt(np.mean((pred-truth)**2))/np.sqrt(np.mean(truth**2));print(n,mode,err,b[0],np.sum(b[1:n+1]>1e-5),scale)
