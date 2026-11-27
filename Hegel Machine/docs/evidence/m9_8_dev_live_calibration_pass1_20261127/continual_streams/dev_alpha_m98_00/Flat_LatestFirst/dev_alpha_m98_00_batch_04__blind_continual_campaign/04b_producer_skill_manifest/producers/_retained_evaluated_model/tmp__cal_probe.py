import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; L=c['true_channel_loads']; R=c['bridge_raw'];
for k,v in c['single_channel_sweeps'].items(): print('sweep',k,v.shape,np.asarray(v).ravel()[:24]);
for name,a in [('cal L',L),('cal R',R),('train hold',D['train']['hold_channel_raw']),('train past',D['train']['past_channel_raw'].reshape(-1,4)),('pilot hold',D['chamber_verification_trials']['hold_channel_raw']),('test hold',D['test']['hold_channel_raw'])]: print(name,np.percentile(a,[0,10,50,90,100],axis=0).round(3));
print('first calibration rows',np.c_[L,R][:12].round(3));
def des(x,kind):
 x=np.atleast_2d(x); cols=[np.ones(len(x)),*x.T];
 if kind in ('diag','quad','cubic','diag3'): cols += list((x*x).T)
 if kind in ('quad','cubic'): cols += [x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)]
 if kind=='cubic': cols += [x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i,4) for k in range(j,4)]
 if kind=='diag3': cols += list((x**3).T)
 return np.column_stack(cols)
for kind in ('affine','diag','quad','diag3','cubic'):
 for fwd in (False,True):
  err=[]
  for k in range(5):
   ix=np.arange(len(R))%5==k; t=~ix; X,Y=(L,R) if fwd else (R,L); b=np.linalg.lstsq(des(X[t],kind),Y[t],rcond=None)[0]; z0=des(R[ix],'affine')@np.linalg.lstsq(des(R[t],'affine'),L[t],rcond=None)[0];
   z=np.array([least_squares(lambda x: (des(x,kind)@b).ravel()-r,a).x for r,a in zip(R[ix],z0)]) if fwd else des(R[ix],kind)@b;err.extend(z-L[ix]);
  e=np.asarray(err); print(kind,'forward' if fwd else 'inverse',np.sqrt(np.mean(e**2,axis=0)).round(4), 'total RMSE/max',np.sqrt(np.mean(e.sum(1)**2)).round(5),np.max(abs(e.sum(1))).round(5));
