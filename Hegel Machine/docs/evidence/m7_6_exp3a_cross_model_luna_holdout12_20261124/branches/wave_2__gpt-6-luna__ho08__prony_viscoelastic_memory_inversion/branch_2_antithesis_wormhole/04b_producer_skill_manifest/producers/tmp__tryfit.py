import pickle,sys,numpy as np
from scipy.optimize import nnls
sys.path.insert(0,'/holdout_task')
from prony_rheology import _memory_design
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb'));e=p['experiment'];t=e['train_t'];x=e['train_strain'];y=e['train_stress_raw'];tx=e['test_t'];xx=e['test_strain'];truth=p['gt_test_stress']
for n in [8,12,20,32,48]:
 taus=np.geomspace(.02,20,n);D=_memory_design(t,x,taus);A=np.c_[x,D,np.ones(len(t)),t];Ax=np.c_[xx,_memory_design(tx,xx,taus)]
 Q=np.c_[np.ones(len(t)),t];P=np.eye(len(t))-Q@np.linalg.pinv(Q);B=P@A[:,:n+1];z=P@y;w=np.ones(len(y))
 for it in range(5):
  sw=np.sqrt(w); norms=np.maximum(np.linalg.norm(B*sw[:,None],axis=0),1e-12);c,_=nnls((B*sw[:,None])/norms,z*sw,maxiter=2000);b=c/norms
  dr=y-A[:,:n+1]@b;d=np.linalg.lstsq(Q,dr,rcond=None)[0];resid=dr-Q@d;s=1.4826*np.median(abs(resid-np.median(resid)))+1e-10;w=np.minimum(1,1.345*s/np.maximum(abs(resid),1e-12))
 pred=Ax@b;err=np.sqrt(np.mean((pred-truth)**2))/np.sqrt(np.mean(truth**2));print(n,err,'Ginf',b[0],flush=True)
