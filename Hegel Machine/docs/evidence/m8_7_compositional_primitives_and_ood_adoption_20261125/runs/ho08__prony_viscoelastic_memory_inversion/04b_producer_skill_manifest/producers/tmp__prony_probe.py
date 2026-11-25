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
for k in ('train','test'):
 t,e=E[k+'_t'],E[k+'_strain']; idx=np.where(abs(np.diff(e,n=2))>.004)[0]+1
 print(k, 'strain range',min(e),max(e),'large curvature',list(zip(t[idx],e[idx]))[:25]);print(np.column_stack([t[::30],e[::30]]))
for mode in ['linear','grad_right','grad_left','grad_trap']:
 T=E['train_t']; e=E['train_strain']; y=E['train_stress_raw']; B=np.column_stack([e,basis(T,e,tau,mode)]); D=np.column_stack([np.ones(T.size),T]); m=np.ones(T.size,dtype=bool)
 for j in range(6):
  q,_=np.linalg.qr(D[m]); bm=B[m]; ym=y[m]; c=nnls(bm-q@(q.T@bm),ym-q@(q.T@ym),maxiter=4000)[0]; d=np.linalg.lstsq(D[m],(y-B@c)[m],rcond=None)[0]; res=y-B@c-D@d; med=np.median(res); sd=1.4826*np.median(abs(res-med)); m=abs(res-med)<4*sd
 tt,ee=E['test_t'],E['test_strain']; F=basis(tt,ee,tau,mode); gt=P['gt_test_stress']; pred=c[0]*ee+F@c[1:]; fixed=gt-ee*P['gt_G_inf']; cc=nnls(F,fixed,maxiter=4000)[0]; print(mode,'G',c[0],'tau/G',list(zip(tau[c[1:]>0],c[1:][c[1:]>0])),'drift',d,'MAD',sd,'kept',m.sum(),'test err',np.linalg.norm(pred-gt)/np.linalg.norm(gt),'oracle err',np.linalg.norm(F@cc-fixed)/np.linalg.norm(gt),'oracle modes',list(zip(tau[cc>0],cc[cc>0])))
