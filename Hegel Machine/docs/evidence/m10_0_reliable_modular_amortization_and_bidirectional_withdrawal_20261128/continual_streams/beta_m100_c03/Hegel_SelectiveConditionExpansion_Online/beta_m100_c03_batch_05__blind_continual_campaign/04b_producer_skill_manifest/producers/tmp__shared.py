import numpy as np,pickle
from scipy.optimize import least_squares
from numpy.polynomial.chebyshev import chebvander
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];c=s['sensor_calibration_sweeps'];d=c['single_bank_sweeps'];t=d['ref_signal'];r=d['raw_readout'];k=d['bank_id'];m=abs(r)<6;x=c['ref_rosette'];y=c['raw_rosette'];g=np.max(abs(y),1)<6
# Anchor gain and offset using one bank, then pool all points on the same amplitude grid.
T=np.unique(t); arr=np.array([r[(t==v)] for v in T]); arr[abs(arr)>6]=np.nan
for j in range(4):
 z=np.isfinite(arr[:,2]+arr[:,j]); b=np.linalg.lstsq(np.c_[np.ones(z.sum()),arr[z,2]],arr[z,j],rcond=None)[0];print('gain,shift',j,b);arr[:,j]=(arr[:,j]-b[0])/b[1]
print('shared curve',np.c_[T,np.nanmean(arr,1)].round(8));
# Compact odd rational shape: fit a numerator and denominator with variable projection, including bank offsets/gains.
for p,q in [(0,1),(1,1),(1,2),(2,2),(2,3),(3,3),(0,3),(1,3),(2,4)]:
 def f(t,a):
  return (t+sum(a[i]*t**(2*i+3) for i in range(p)))/(1+sum(a[p+i]*t**(2*i+2) for i in range(q)))
 def fit(a):
  h=np.c_[np.ones(len(t)),f(t,a)];B=np.array([np.linalg.lstsq(h[m&(k==j)],r[m&(k==j)],rcond=None)[0] for j in range(4)]);return (np.sum(h*B[k],1)-r)[m],B
 opt=least_squares(lambda a:fit(a)[0],np.r_[np.full(p,.02),.6,np.full(q-1,.02)],bounds=(np.r_[np.full(p,-1.),np.zeros(q)],np.full(p+q,20.)),max_nfev=180,gtol=1e-12);e,B=fit(opt.x);v=f(x,opt.x);print('law',p,q,opt.x.round(6),'rms',np.mean(e**2)**.5,'diag/offset',B.round(6).tolist())
 for label,w in [('v+x2',np.c_[v,x*x]),('v+v2',np.c_[v,v*v]),('v+x2+v2',np.c_[v,x*x,v*v]),('v+x+x2',np.c_[v,x,x*x]),('v+x+x2+v2',np.c_[v,x,x*x,v*v])]:
  H=np.c_[np.ones(len(x)),w]; bb=np.linalg.lstsq(H[g],y[g],rcond=None)[0];E=(H@bb-y)[g];print(label,np.sqrt(np.mean(E*E,0)).round(6),round(np.max(abs(E)),6));
  if p==2 and q==3 and label=='v+x+x2+v2':print(bb.round(6).T)
