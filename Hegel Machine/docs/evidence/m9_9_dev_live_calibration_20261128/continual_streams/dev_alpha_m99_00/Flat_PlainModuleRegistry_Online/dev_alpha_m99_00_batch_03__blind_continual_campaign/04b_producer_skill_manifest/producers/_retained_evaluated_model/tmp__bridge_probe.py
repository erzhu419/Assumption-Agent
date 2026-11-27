import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; c=p['dataset']['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; ax=(x!=0).sum(1)<=1; spikes=np.max(abs(y),axis=1)>6
print('axis rows',np.where(ax)[0], 'provisional spikes',np.where(spikes)[0], 'max provisionally clean',np.max(abs(y[~spikes])))
for degree in range(1,5):
 terms=[t for j in range(degree+1) for t in combinations_with_replacement(range(4),j)]
 a=np.column_stack([np.prod(x[:,t],axis=1) for t in terms]); err=np.full_like(y,np.nan)
 for i in np.where(~spikes)[0]:
  tr=~spikes & (np.arange(len(x))!=i)
  b=np.linalg.lstsq(a[tr],y[tr],rcond=None)[0]; err[i]=a[i]@b-y[i]
 print('degree',degree,'rank',np.linalg.matrix_rank(a[~spikes]),'cols',len(terms),'axis/mixed RMSE, max',[(np.sqrt(np.nanmean(err[z]**2,axis=0)).round(6).tolist(),round(np.nanmax(abs(err[z])),6)) for z in (ax,~ax)])
 if degree==3:
  b=np.linalg.lstsq(a[~spikes],y[~spikes],rcond=None)[0]; print('cubic leading coefficients:', [[(terms[j],round(b[j,k],5)) for j in np.argsort(abs(b[:,k]))[-10:][::-1]] for k in range(4)])
  m=np.max(abs(a@b-y),axis=1); print('largest residuals:',[(int(i),round(m[i],4)) for i in np.argsort(m)[-14:][::-1]])
print('axis clean data',np.round(np.c_[x,y][ax & ~spikes],5))
