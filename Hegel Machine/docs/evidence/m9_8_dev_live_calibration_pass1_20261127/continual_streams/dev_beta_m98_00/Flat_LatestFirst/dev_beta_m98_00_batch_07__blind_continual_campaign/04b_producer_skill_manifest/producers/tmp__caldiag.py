import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; b=c['single_bank_sweeps']; I=np.eye(4); xb=I[b['bank_id']]*b['ref_signal'][:,None]; yb=b['raw_readout']; np.set_printoptions(precision=5,suppress=True,linewidth=150)
def basis(t,d):
 ids=[i for k in range(d+1) for i in combinations_with_replacement(range(4),k)]; return np.column_stack([np.prod(t[:,i],axis=1) for i in ids]), ids
for j in range(4):
 A,ids=basis(x,3); beta=least_squares(lambda p:A@p-y[:,j], np.linalg.lstsq(A,y[:,j],rcond=None)[0],loss='cauchy',f_scale=.03,max_nfev=400).x; e=y[:,j]-A@beta
 print('channel',j,'gross',[(i,round(float(e[i]),3)) for i in np.flatnonzero(abs(e)>3)], 'cubic coeffs',list(zip(ids,np.round(beta,4))))
 print('reference axis rows',np.c_[x[:16],y[:16,j],e[:16]]); q=b['bank_id']==j; print('bank residual',yb[q]-basis(xb[q],3)[0]@beta)
eq=np.max(np.abs(y-x),axis=1)>8
print('gross rows',np.where(eq)[0],np.c_[x[eq],y[eq]])
for d in (1,2,3):
 A,_=basis(x,d); err=[]
 for k in range(5):
  tr=(np.arange(len(x))%5!=k)&~eq; te=(np.arange(len(x))%5==k)&~eq
  err.append(np.sqrt(np.mean((A[te]@np.linalg.lstsq(A[tr],y[tr],rcond=None)[0]-y[te])**2,axis=0)))
 print('gross-excursion-screened polynomial',d,'max fold by channel',np.max(err,axis=0))
