import pickle
import numpy as np
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; n=len(x)
np.set_printoptions(precision=5,suppress=True,linewidth=125)
print('sweep nonzero count',np.unique(np.count_nonzero(x,axis=1),return_counts=True)); print('raw observations range',s['y_obs_raw'].reshape(-1,4).min(0),s['y_obs_raw'].reshape(-1,4).max(0))
def design(x,kind):
 a=[np.ones(len(x)),*x.T]
 if kind in ('self23','pair2','all3'): a.extend([*(x*x).T,*(x**3).T])
 if kind in ('pair2','all3'): a.extend(x[:,i]*x[:,j] for i,j in combinations(range(4),2))
 if kind=='all3':
  a.extend(x[:,i]**2*x[:,j] for i in range(4) for j in range(4) if i!=j)
  a.extend(x[:,i]*x[:,j]*x[:,k] for i,j,k in combinations(range(4),3))
 return np.array(a).T
def fit(a,y):
 b=np.linalg.lstsq(a,y,rcond=None)[0]
 for _ in range(35):
  r=y-a@b; r=r-np.median(r,axis=0); sc=np.maximum(1.4826*np.median(abs(r),axis=0),1e-7)
  for j in range(4):
   w=(1+(r[:,j]/(2.5*sc[j]))**2)**-.5
   b[:,j]=np.linalg.lstsq(a*w[:,None],y[:,j]*w,rcond=None)[0]
 return b
if __name__=='__main__':
 for kind in ['affine','self23','pair2','all3']:
  a=design(x,kind); b=fit(a,y); e=np.empty_like(y)
  for f in range(5):
   test=(np.arange(n)%5==f); e[test]=y[test]-a[test]@fit(a[~test],y[~test])
  print(kind,'rank',np.linalg.matrix_rank(a),'OOF abs error median and p90 by channel',np.quantile(abs(e),[.5,.9],axis=0),'large residual indices',np.where(np.max(abs(e),axis=1)>.3)[0]); print('coefficients',b if kind!='all3' else np.linalg.norm(b,axis=1))
 print('pure sweeps:'); print(np.column_stack((x[:16],y[:16])))
