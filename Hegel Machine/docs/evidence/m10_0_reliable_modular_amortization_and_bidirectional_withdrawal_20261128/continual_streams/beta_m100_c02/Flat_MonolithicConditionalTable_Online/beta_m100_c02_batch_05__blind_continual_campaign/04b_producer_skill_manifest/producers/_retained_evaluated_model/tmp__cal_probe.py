import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.stats import spearmanr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; X,Y=a['ref_rosette'],a['raw_rosette']; b=a['single_bank_sweeps']; np.set_printoptions(precision=4,suppress=True,linewidth=120)
print('ranges',X.min(0),X.max(0),np.quantile(Y,[0,.05,.5,.95,1],axis=0)); print('ranks',spearmanr(X,Y).statistic[:4,4:]); print('spikes',[(i,row.tolist()) for i,row in enumerate(Y) if np.max(abs(row))>5]);
for i in range(4):
 j=b['bank_id']==i; print('bank',i,np.array([b['ref_signal'][j],b['raw_readout'][j]]).T)
for d in (1,2,3,4):
 terms=[t for k in range(d+1) for t in combinations_with_replacement(range(4),k)]; A=np.stack([np.prod(X[:,t],axis=1) for t in terms],axis=1)
 for j in range(4):
  keep=abs(Y[:,j])<5; c=np.linalg.lstsq(A[keep],Y[keep,j],rcond=None)[0]; err=Y[:,j]-A@c
  print(d,j,'inlier residual',np.quantile(abs(err[keep]),[.5,.95,1]),'largest',[(t,round(v,5)) for t,v in sorted(zip(terms,c),key=lambda tv:-abs(tv[1]))[:9]])
