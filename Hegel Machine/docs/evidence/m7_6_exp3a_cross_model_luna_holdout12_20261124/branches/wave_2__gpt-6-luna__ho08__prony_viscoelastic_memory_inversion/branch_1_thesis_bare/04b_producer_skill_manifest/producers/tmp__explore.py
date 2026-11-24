import pickle,sys,numpy as np
from scipy.optimize import nnls
sys.path.insert(0,'/holdout_task'); import prony_rheology as pr
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']; tt=e['train_t']; eps=e['train_strain']; y=e['train_stress_raw']; te=e['test_t']; et=e['test_strain']; truth=p['gt_test_stress']; D=np.column_stack((np.ones_like(tt),tt-tt.mean())); Q=np.linalg.qr(D,mode='reduced')[0]
for top in [5,8,10,15,20,30,50,100]:
 for n in [24,40]:
  taus=np.geomspace(.02,top,n); mem=np.column_stack([pr._memory_column(tt,eps,tau) for tau in taus]); X=np.column_stack((eps,mem)); Xp=X-Q@(Q.T@X); yp=y-Q@(Q.T@y); w=np.ones_like(y)
  for _ in range(10):
   c=nnls(Xp*np.sqrt(w)[:,None],yp*np.sqrt(w),maxiter=1000)[0]; r=y-X@c; drift=np.linalg.lstsq(D,r,rcond=None)[0]; r-=D@drift; scale=1.4826*np.median(abs(r-np.median(r)))+1e-12; nw=np.minimum(1,1.5*scale/np.maximum(abs(r-np.median(r)),1e-12))
   if np.max(abs(nw-w))<1e-4: break
   w=nw
  mt=np.column_stack([pr._memory_column(te,et,tau) for tau in taus]); pred=c[0]*et+mt@c[1:]; err=np.sqrt(np.mean((pred-truth)**2))/np.sqrt(np.mean(truth**2)); print(top,n,c[0],err,flush=True)
