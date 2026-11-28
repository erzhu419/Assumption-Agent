import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.stats import pearsonr
from scipy.optimize import least_squares
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; R=c['bridge_raw']; Y=c['true_channel_loads']; s=c['single_channel_sweeps']; print('first paired rows',np.c_[Y,R][:10].round(4)); print('ranges Y,R', Y.min(0),Y.max(0),R.min(0),R.max(0));
for j in np.unique(s['lab_id']):
 ix=s['lab_id']==j; x=s['true_stress'][ix]; y=s['cmd_stress'][ix]; p=np.polynomial.polynomial.polyfit(x,y,3); print('sweep',j,p,'rms',np.std(np.polynomial.polynomial.polyval(x,p)-y),'rms degree 1/2',[np.std(np.polynomial.polynomial.polyval(x,np.polynomial.polynomial.polyfit(x,y,k))-y) for k in (1,2)])
def cols(a,inds):return np.column_stack([np.prod(a[:,i],axis=1) for i in inds])
def check(A,B,kind):
 for p in (1,2,3,4,5):
  for diag in ([True,False] if 1<p<4 else [True]):
   inds=[()]+[i for k in range(1,p+1) for i in combinations_with_replacement(range(4),k) if not diag or len(set(i))==1]; X=cols(A,inds); F=np.linalg.lstsq(X,B,rcond=None)[0]; E=X@F-B; H=np.sum(np.linalg.qr(X,mode='reduced')[0]**2,axis=1); print(kind,p,diag,'rms',np.std(E,0).round(6),'LOO',np.std(E/(1-H[:,None]),0).round(6), 'sum rms/LOO',np.std(E.sum(1)),np.std(E.sum(1)/(1-H))); 
   if p==3 and diag:print('coef by degree and input',F.round(5))
check(Y,R,'forward'); check(R,Y,'inverse')
for split in ('train','test'):
 a=np.concatenate([d[split]['past_channel_raw'].reshape(-1,4),d[split]['hold_channel_raw']]);print(split,'raw range',a.min(0),a.max(0))
