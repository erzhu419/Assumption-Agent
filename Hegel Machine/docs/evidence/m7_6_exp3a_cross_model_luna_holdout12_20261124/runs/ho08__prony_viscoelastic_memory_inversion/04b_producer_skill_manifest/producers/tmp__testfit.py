import pickle,sys,numpy as np
from scipy.optimize import nnls
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment'];sys.path.insert(0,'/holdout_task');import prony_rheology as pr
for n in [24,40,64,96]:
 taus=np.geomspace(.02,20,n);t=e['train_t'];x=e['train_strain'];y=e['train_stress_raw']; mem=np.column_stack([pr._memory_column(t,x,q) for q in taus]); X=np.column_stack([x,mem]); D=np.column_stack([np.ones(len(t)),t-t.mean()]);Q=np.linalg.qr(D)[0];Xp=X-Q@(Q.T@X);yp=y-Q@(Q.T@y);w=np.ones(len(y))
 for _ in range(8):
  sw=np.sqrt(w); c=nnls(Xp*sw[:,None],yp*sw,maxiter=500)[0]; r=y-X@c; dc=np.linalg.lstsq(D,r,rcond=None)[0]; r-=D@dc; scale=1.4826*np.median(abs(r-np.median(r)))+1e-12; cut=1.5*scale; nw=np.minimum(1,cut/np.maximum(abs(r-np.median(r)),1e-12))
  if np.max(abs(nw-w))<1e-4:break
  w=nw
 mt=np.column_stack([pr._memory_column(e['test_t'],e['test_strain'],q) for q in taus]);pred=c[0]*e['test_strain']+mt@c[1:];truth=p['gt_test_stress'];print(n,c[0],np.sqrt(np.mean((pred-truth)**2))/np.sqrt(np.mean(truth**2)),scale,flush=True)
