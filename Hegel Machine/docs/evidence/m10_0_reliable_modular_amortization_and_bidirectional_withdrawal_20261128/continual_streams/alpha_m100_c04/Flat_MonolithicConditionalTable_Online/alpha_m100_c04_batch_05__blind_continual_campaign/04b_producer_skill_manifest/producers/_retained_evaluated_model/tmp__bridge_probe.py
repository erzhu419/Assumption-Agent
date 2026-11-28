import pickle,numpy as np
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; q0=c['true_channel_loads']; r0=c['bridge_raw']; m=np.max(abs(r0),axis=1)<8; q=q0[m]; r=r0[m]
np.set_printoptions(precision=6,suppress=True)
for j in range(4):
 print('AXIS',j,np.c_[q0[j*4:j*4+4,j],r0[j*4:j*4+4]])
odd={'tanh':np.tanh,'atan':np.arctan,'rational':lambda u:u/(1+u*u),'sqrt':lambda u:u/np.sqrt(1+u*u),'asinh':np.arcsinh,'softsign':lambda u:u/(1+abs(u)),'sin':np.sin,'identity':lambda u:u}
even={'abs':lambda u,k:abs(u),'q*tanh':lambda u,k:u*np.tanh(k*u),'logcosh':lambda u,k:np.logaddexp(k*u,-k*u)-np.log(2),'square':lambda u,k:u*u,'sat_square':lambda u,k:u*u/(1+k*u*u),'abs_qsat':lambda u,k:abs(u)*(1-np.exp(-k*abs(u)))}
axis=(q!=0).sum(1)==1
for j in range(4):
 u=q[:,j]; v=q[:,(j+2)%4]; n=q[:,(j+1)%4]; y=r[:,j]; results=[]
 for ok,of in odd.items():
  for ek,ef in even.items():
   def A(p):
    return np.column_stack([np.ones(len(u)),of(p[0]*u),v,ef(v,p[1]),u*n])
   def fit(mask):
    def calc(p):
     a=A(p)[mask]; w=np.linalg.lstsq(a,y[mask],rcond=None)[0]; return a@w-y[mask]
    f=least_squares(calc,[1,2],bounds=([.01,.01],[10,100]),max_nfev=100)
    a=A(f.x); w=np.linalg.lstsq(a[mask],y[mask],rcond=None)[0]; return f.x,w,a@w-y
   p,w,e=fit(np.ones(len(u),bool)); results.append((np.sqrt(np.mean(e**2)),ok,ek,p,w,np.sqrt(np.mean(e[axis]**2))))
 print('FIT',j,*sorted(results,key=lambda z:z[0])[:8],sep='\n')
