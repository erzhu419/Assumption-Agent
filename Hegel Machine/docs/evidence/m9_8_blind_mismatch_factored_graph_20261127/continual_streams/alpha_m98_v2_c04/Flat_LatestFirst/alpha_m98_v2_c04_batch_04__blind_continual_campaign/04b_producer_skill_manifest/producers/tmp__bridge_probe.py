import pickle,numpy as np
from itertools import combinations,combinations_with_replacement
np.set_printoptions(precision=6,suppress=True)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset'];c=d['calibration_load_cells'];x=c['true_channel_loads'];y=c['bridge_raw']; good=np.max(abs(y),axis=1)<8
xs=x[good];ys=y[good]; n=len(xs); pairs=list(combinations(range(4),2));
base={'linear':lambda a:a,'sq':lambda a:a*a,'signed_sq':lambda a:a*abs(a),'cube':lambda a:a**3,'pair':lambda a:np.column_stack([a[:,j]*a[:,k] for j,k in pairs]),'triple':lambda a:np.column_stack([a[:,p].prod(1) for p in combinations(range(4),3)]),'mixedcube':lambda a:np.column_stack([a[:,j]**2*a[:,k] for j in range(4) for k in range(4) if j!=k])}
def design(a,terms):return np.column_stack([np.ones(len(a))]+[base[k](a) for k in terms.split()]);
for terms in ['linear','linear sq','linear signed_sq','linear cube','linear cube pair','linear signed_sq pair','linear sq pair','linear sq cube pair','linear cube pair triple','linear cube pair mixedcube']:
 A=design(xs,terms); beta=np.linalg.lstsq(A,ys,rcond=None)[0]; E=ys-A@beta; L=np.sum(np.linalg.qr(A)[0]**2,axis=1); ee=E/(1-L[:,None]);print(terms,'LOO RMSE',np.sqrt(np.mean(ee**2,axis=0)),'fit RMSE',np.sqrt(np.mean(E**2,axis=0)))
 if terms in ('linear cube pair','linear signed_sq pair','linear sq cube pair'):print('coef',beta)
 for i in range(4):
  h=(np.count_nonzero(xs,axis=1)==1)&(xs[:,i]!=0)
  if terms in ('linear signed_sq pair','linear cube pair'):print('axis',i,'diagonal residual',E[h,i]);
