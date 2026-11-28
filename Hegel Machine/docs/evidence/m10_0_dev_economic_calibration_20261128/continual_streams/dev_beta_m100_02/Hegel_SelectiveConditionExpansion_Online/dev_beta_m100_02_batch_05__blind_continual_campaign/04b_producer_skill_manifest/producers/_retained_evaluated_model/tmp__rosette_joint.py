import pickle, numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; bias=np.array([-.06,.07,.06,-.05]); n=len(x); clean=(abs(y-bias).max(1)<5); mixed=(np.count_nonzero(x,axis=1)>1)&clean
print('counts',n,clean.sum(),mixed.sum()); print('mix ranges, abs obs q',x[mixed].min(0),x[mixed].max(0),np.quantile(abs(s['y_obs_raw']),[.5,.95,1])); print('axis off diagonal relative to bias')
for i in range(4):
 m=(x[:,i]!=0)&(np.count_nonzero(x,axis=1)==1)&clean; print(i,np.round(np.c_[x[m,i],y[m]-bias],7))
D=[]
for j in range(4):
 m=(b['bank_id']==j)&(abs(b['raw_readout']-bias[j])<5); u=b['ref_signal'][m]; v=b['raw_readout'][m]-bias[j];
 q=least_squares(lambda p:p[0]*u+p[1]*np.sin(p[2]*u)-v,[.8,.2,2.5]); print('sin fit',j,q.x,np.max(abs(q.fun))); D.append(q.x)
D=np.array(D)
def direct(x):return D[:,0]*x+D[:,1]*np.sin(D[:,2]*x)
r=y-bias-direct(x); print('mixed residual q',np.quantile(r[mixed],[0,.25,.5,.75,1],axis=0)); print('sample mixed residuals',np.round(np.c_[x[mixed],r[mixed]][:12],5)); print('telemetry',s['aux_manifold_telemetry'][:3])
from itertools import combinations_with_replacement
for deg in range(1,7):
 idx=[I for k in range(1,deg+1) for I in combinations_with_replacement(range(4),k) if len(set(I))<=2]; X=np.column_stack([np.ones(n)]+[x[:,I].prod(1) for I in idx]); axes=np.count_nonzero(x,axis=1)==1
 def fit(tr,va):
  p=np.linalg.lstsq(X[tr],r[tr],rcond=1e-8)[0]; e=(X@p-r)[va]; return np.sqrt(np.mean(e**2)),np.max(abs(e))
 ids=np.flatnonzero(mixed); rng=np.random.default_rng(0); folds=np.array_split(rng.permutation(ids),5); er=[fit(clean&~np.isin(np.arange(n),v),v) for v in folds]; edge=ids[np.max(abs(x[ids]),1)>1.35]; print('poly',deg,len(idx),fit(clean,clean),np.round(np.max(er,axis=0),7),'edge',fit(clean&~np.isin(np.arange(n),edge),edge))
