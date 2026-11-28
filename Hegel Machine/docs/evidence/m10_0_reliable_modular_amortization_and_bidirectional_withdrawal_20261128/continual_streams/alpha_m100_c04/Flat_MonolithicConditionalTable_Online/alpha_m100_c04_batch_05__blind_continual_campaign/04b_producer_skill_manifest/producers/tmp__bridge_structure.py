import pickle,numpy as np
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; q=c['true_channel_loads']; r=c['bridge_raw']; m=np.max(abs(r),axis=1)<8; q=q[m]; r=r[m]; ax=(q!=0).sum(1)==1
print('mixed load min/max',q[~ax].min(0),q[~ax].max(0))
odd={'atan':np.arctan,'tanh':np.tanh,'erf':__import__('scipy.special',fromlist=['erf']).erf,'sqrt':lambda z:z/np.sqrt(1+z*z),'rational':lambda z:z/(1+z*z),'sin':np.sin}
even={'sat_square':lambda v,k:v*v/(1+k*v*v),'tanh_square':lambda v,k:np.tanh(k*v)**2,'q_tanh':lambda v,k:v*np.tanh(k*v),'logcosh':lambda v,k:np.logaddexp(k*v,-k*v)-np.log(2),'one_exp':lambda v,k:-np.expm1(-k*v*v),'sqrt':lambda v,k:np.sqrt(1+k*v*v)-1}
for j in range(4):
 u=q[:,j]; v=q[:,(j+2)%4]; y=r[:,j]; results=[]
 for on,of in odd.items():
  for en,ef in even.items():
   for ot in ('none','atan','tanh','sat_cube'):
    def a(p):
     extra={'none':(), 'atan':(np.arctan(p[2]*v),), 'tanh':(np.tanh(p[2]*v),), 'sat_cube':(v**3/(1+p[2]*v*v),)}[ot]
     return np.column_stack((np.ones(len(u)),u,of(p[0]*u),v,ef(v,p[1]),*extra))
    def fit(m):
     def residual(p):
      A=a(p)[m]; return A@np.linalg.lstsq(A,y[m],rcond=None)[0]-y[m]
     p=least_squares(residual,[1.25,1.,1.],bounds=([.05,.03,.03],[4,12,8]),max_nfev=55).x
     A=a(p); w=np.linalg.lstsq(A[m],y[m],rcond=None)[0]; return p,w,A@w-y
    p,w,e=fit(np.ones(len(u),bool)); results.append((np.sqrt(np.mean(e**2)),on,en,ot,p.round(5),w.round(5),np.max(abs(e)),np.sqrt(np.mean(e[ax]**2))))
 print('CHAN',j,*sorted(results,key=lambda z:z[0])[:6],sep='\n')
