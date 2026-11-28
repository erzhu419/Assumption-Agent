import numpy as np, pickle
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; bank=c['single_bank_sweeps']; n=len(x)
mask=np.max(abs(y-np.median(y,0))/np.median(abs(y-np.median(y,0)),0),1)<9
for j in range(4):
 z=x[:,j]; p=x[:,(j+2)%4]; isb=(bank['bank_id']==j)&(abs(bank['raw_readout'])<3); u=bank['ref_signal'][isb]; v=bank['raw_readout'][isb];
 def solve(a):
  D=np.c_[np.ones(n),np.tanh(a*z),p,p*p]; B=np.c_[np.ones(len(u)),np.tanh(a*u),np.zeros((len(u),2))]; coef=np.linalg.lstsq(np.r_[D[mask],B],np.r_[y[mask,j],v],rcond=None)[0]; e=y[:,j]-D@coef; eb=v-B@coef; return np.sum(e[mask]**2)+np.sum(eb**2),coef,e,eb
 a=minimize_scalar(lambda a:solve(a)[0],bounds=(.3,3),method='bounded').x; _,coef,e,eb=solve(a)
 print('channel',j,'scale/coeffs',a,coef,'sweep/bank RMS',np.sqrt(np.mean(e[mask]**2)),np.sqrt(np.mean(eb**2))); print('worst',[(i,round(e[i],4),np.round(x[i],3).tolist()) for i in sorted(np.flatnonzero(mask),key=lambda i:-abs(e[i]))[:8]]); print('residual correlations x and x2',np.round(np.corrcoef(np.c_[e[mask],x[mask],x[mask]**2].T)[0,1:],3)); print('bank residuals',np.round(eb,4))
print('row residual matrices for fixed structured tanh:')
r=[]
for j in range(4):
 p=x[:,(j+2)%4]; A=np.c_[np.ones(n),np.tanh(1.1*x[:,j]),p,p*p]; b=np.linalg.lstsq(A[mask],y[mask,j],rcond=None)[0]; print(j,np.round(b,6)); r.append(y[:,j]-A@b)
r=np.array(r).T
for i in np.flatnonzero(mask): print(i,*np.round(r[i],4))
