import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,q=(s['train'][k] for k in ('x','y','traj')); z,L,M=m._features(s,x); idx=np.arange(len(x)); t=s['eval_times']; dot=M[:,[0,5,10,15]].sum(1); D=np.c_[L,dot]; F=-np.expm1(-t[:,None]*[1.8,.4]); slow=np.linalg.lstsq(F,q.T,rcond=None)[0][1]
big=np.argsort(np.linalg.norm(z,axis=1))[-14:]; extreme=np.argsort(abs(slow))[-12:]; d=np.linalg.norm((z[:,None,:]-z[None,:,:]),axis=-1); np.fill_diagonal(d,np.inf); isolated=np.argsort(d.min(1))[-12:]; folds=[idx%6==k for k in range(6)]+[idx>=56,np.isin(idx,big),np.isin(idx,extreme),np.isin(idx,isolated)]
errs=[]
for va in folds:
 st=dict(s,train=dict(s['train'],x=x[~va],y=y[~va],traj=q[~va]),test={'x':x[va]}); out=m.solve_batch(st); errs.append(np.r_[abs(out['y_hat']-y[va]).mean(),abs(out['state_pred']-q[va]).mean(0)])
E=np.array(errs);print('artifact regular worst [y,traj]',E[:6,0].max(),E[:6,1:].max(),'stress [y,traj]',np.c_[E[6:,0],E[6:,1:].max(1)])
for fitrates in [False, True]:
 for name,S in [('all',np.c_[L,M]),('abf',np.c_[L,M[:,1]]),('three',np.c_[L,M[:,[1,7,14]]]),('three dot',np.c_[D,M[:,[1,7,14]]]),('three diag',np.c_[L,M[:,[0,1,5,7,10,14,15]]]),('six',np.c_[L,M[:,[0,1,7,8,14,15]]])]:
  ee=[]; rr=[]
  for va in folds:
   tr=~va; qtr=q[tr]; fr=F
   if fitrates:
    def ef(p):
     f=-np.expm1(-t[:,None]*p);return (f@np.linalg.lstsq(f,qtr.T,rcond=None)[0]-qtr.T).ravel()
    p=least_squares(ef,[1.8,.4],bounds=(.01,10)).x;fr=-np.expm1(-t[:,None]*p)
   Tar=np.linalg.lstsq(fr,qtr.T,rcond=None)[0].T; f=D[va]@m._ridge(D[tr],Tar[:,0],.00015); a=S[va]@m._ridge(S[tr],Tar[:,1],.0003);ee.append(abs(np.c_[f,a]@fr.T-q[va]).mean(0)); rr.append(np.sqrt(np.mean((S[tr]@m._ridge(S[tr],Tar[:,1],.0003)-Tar[:,1])**2)))
  e=np.array(ee);print(fitrates,name,'rms',round(np.mean(rr),5),'regular',np.round([e[:6].mean(),e[:6].max()],5),'stress',np.round(e[6:].max(1),5))
