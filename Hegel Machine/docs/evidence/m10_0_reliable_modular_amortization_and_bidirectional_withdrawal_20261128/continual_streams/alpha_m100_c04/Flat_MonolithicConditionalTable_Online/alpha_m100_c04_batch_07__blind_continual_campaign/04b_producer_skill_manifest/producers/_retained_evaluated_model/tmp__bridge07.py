import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; l=c['true_channel_loads']; y=c['bridge_raw']; print('ranges', l.min(0),l.max(0),y.min(0),y.max(0)); print('sweep',[(k,np.asarray(v)[:12]) for k,v in c['single_channel_sweeps'].items()]);
def terms(l,d):
 return np.column_stack([np.prod(l[:,ix],axis=1) for k in range(d+1) for ix in combinations_with_replacement(range(4),k)])
def fit(A,y):
 p=np.linalg.lstsq(A,y,rcond=None)[0]
 for k in range(12):
  r=A@p-y; s=max(.0004,1.4826*np.median(abs(r-np.median(r)))); w=1/np.sqrt(1+(r/(3*s))**2);p=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0]
 return p
for degree in range(1,5):
 A=terms(l,degree); r=np.zeros_like(y); pc=[]
 for j in range(4):
  pc.append(fit(A,y[:,j]));
  for sel in np.array_split(np.argsort(np.max(abs(l),axis=1)),8):
   m=np.ones(len(y),bool);m[sel]=False; r[sel,j]=A[sel]@fit(A[m],y[m,j])-y[sel,j]
 print(degree,'quantiles',np.quantile(abs(r),[.5,.9,.95,1],axis=0).round(5),'outer-block medians',[round(float(np.median(abs(r[sel]))),5) for sel in np.array_split(np.argsort(np.max(abs(l),axis=1)),8)]); print('coefs',np.array(pc).round(4).T.tolist())
