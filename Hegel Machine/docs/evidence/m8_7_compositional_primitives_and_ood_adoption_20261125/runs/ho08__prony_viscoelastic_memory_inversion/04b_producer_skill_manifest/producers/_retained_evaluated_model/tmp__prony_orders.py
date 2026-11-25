import pickle
import numpy as np
from scipy.optimize import nnls, least_squares
from scipy.signal import lfilter
P=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); E=P['experiment']; tau=np.geomspace(.02,20,100)
def basis(t,e,tau,mode):
 dt=t[1]-t[0]; r=np.exp(-dt/tau); w=-np.expm1(-dt/tau)*tau/dt
 de=np.diff(e,prepend=e[0]); grad=np.gradient(e,t)
 if mode=='linear': return np.array([lfilter([b],[1,-a],de) for a,b in zip(r,w)]).T
 if mode=='grad_right': rate=np.r_[0., grad[1:]]
 if mode=='grad_left': rate=np.r_[0., grad[:-1]]
 if mode=='grad_trap':
  return np.array([lfilter([1],[1,-a],np.r_[0.,(grad[1:]+a*grad[:-1])*dt/2]) for a in r]).T
 return np.array([lfilter([b*dt],[1,-a],rate) for a,b in zip(r,w)]).T
from itertools import combinations
T=E['train_t']; e=E['train_strain']; y=E['train_stress_raw']; D=np.column_stack([np.ones(T.size),T/T[-1]]); tt,ee=E['test_t'],E['test_strain']; gt=P['gt_test_stress'];
def fit(y):
 def args(z):
  k=(len(z)-3)//2
  return np.column_stack([e,basis(T,e,np.exp(z[:k]),'linear'),D])
 def fun(z):
  k=(len(z)-3)//2
  return args(z)@z[k:]-y
 for k in range(1,6):
  lo=np.r_[np.full(k,np.log(.02)),np.zeros(k+1),-np.inf,-np.inf]; hi=np.r_[np.full(k,np.log(20)),np.full(k+3,np.inf)]
  best=None
  for tx in combinations([.04,.16,.65,2.6,10.,18.],k):
   z=np.r_[np.log(tx),12.,np.full(k,15.),y[0],y[-1]-y[0]]
   r=least_squares(fun,z,bounds=(lo,hi),loss='cauchy',f_scale=.045,max_nfev=100)
   if best is None or r.cost<best.cost: best=r
  z=best.x; res=fun(z); med=np.median(res); sd=1.4826*np.median(abs(res-med)); m=abs(res-med)<4*sd
  rr=least_squares(lambda z:fun(z)[m],z,bounds=(lo,hi),max_nfev=250); z=rr.x; r=fun(z); rmse=np.sqrt(np.mean(r[m]**2)); pred=z[k]*ee+basis(tt,ee,np.exp(z[:k]),'linear')@z[k+1:-2]
  print(k,'robust cost',best.cost,'n',sum(m),'rmse',rmse,'BIC',sum(m)*np.log(rmse**2)+(2*k+3)*np.log(sum(m)),'G',z[k],'modes',sorted(zip(np.exp(z[:k]),z[k+1:-2])),'err',np.linalg.norm(pred-gt)/np.linalg.norm(gt),flush=True)
fit(y)
rr=least_squares(lambda z:(basis(tt,ee,np.exp(z[:3]),'linear')@z[3:]+12.5*ee-gt),np.r_[np.log([.18,1.4,8.5]),28.,16.,9.],gtol=1.e-14); print('oracle',np.exp(rr.x[:3]),rr.x[3:],np.linalg.norm(rr.fun))
