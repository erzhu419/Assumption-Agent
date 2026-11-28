import pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; y=p['true_channel_loads']; raw=p['bridge_raw']; clean=np.max(np.abs(raw),1)<6
np.set_printoptions(precision=6,suppress=True)
print('excluded rows',np.flatnonzero(~clean)); groups=np.count_nonzero(y,axis=1)
pairs=list(combinations(range(4),2))
def basis(y,j,form):
 a=[np.ones(len(y)),*y.T]
 if form>=1:a += [y[:,j]**3]
 if form>=2:a += [y[:,j]**2]
 if form>=3:a += [y[:,k]**3 for k in range(4) if k!=j] + [y[:,k]**2 for k in range(4) if k!=j]
 if form>=4:a += [y[:,k]*y[:,l] for k,l in pairs]
 return np.column_stack(a)
for f in range(5):
 print('form',f)
 for j in range(4):
  A=basis(y,j,f); b=raw[:,j]; errs=[]
  for idx in np.flatnonzero(clean):
   keep=clean.copy();keep[idx]=False; q=np.linalg.lstsq(A[keep],b[keep],rcond=None)[0];errs.append(A[idx]@q-b[idx])
  e=np.array(errs); q=np.linalg.lstsq(A[clean],b[clean],rcond=None)[0];print(j,'LOO max, rmse',np.max(abs(e)),np.sqrt(np.mean(e**2)), 'coef',q)
s=p['single_channel_sweeps']; print('sweep holdout errors from diagonal cubic')
for j in range(4):
 sel=s['lab_id']==j; x=s['true_stress'][sel]; pred=np.zeros((len(x),4));pred[:,j]=x
 A=basis(y,j,1); q=np.linalg.lstsq(A[clean],raw[clean,j],rcond=None)[0]; err=basis(pred,j,1)@q-s['cmd_stress'][sel]; print(j,err)
