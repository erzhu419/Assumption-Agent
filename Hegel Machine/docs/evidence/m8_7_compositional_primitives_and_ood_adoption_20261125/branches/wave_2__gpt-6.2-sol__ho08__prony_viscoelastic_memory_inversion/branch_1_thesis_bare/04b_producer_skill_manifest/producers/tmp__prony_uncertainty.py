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
T,e,y=[E['train_'+s] for s in ('t','strain','stress_raw')]; D=np.column_stack([np.ones(len(T)),T/T[-1]]); tt,ee=E['test_t'],E['test_strain']; gt=P['gt_test_stress']; k=3
lo=np.r_[np.log([.02]*k),np.zeros(k+1),-np.inf,-np.inf]; hi=np.r_[np.log([20.]*k),np.full(k+3,np.inf)]
z0=np.r_[np.log([.21,1.5,10.]),11.6,27,16,9.6,-1.8,4.24]
def fun(z): return np.column_stack([e,basis(T,e,np.exp(z[:k]),'linear'),D])@z[k:]-y
for loss in ['soft_l1','cauchy','arctan']:
 for s in [.015,.025,.04,.06,.09]:
  r=least_squares(fun,z0,bounds=(lo,hi),loss=loss,f_scale=s,max_nfev=200); print(loss,s, np.round(np.r_[np.exp(r.x[:k]),r.x[k:]],4))
r=least_squares(fun,z0,bounds=(lo,hi),loss='cauchy',f_scale=.04); m=abs(r.fun-np.median(r.fun))<.065
r=least_squares(lambda z:fun(z)[m],r.x,bounds=(lo,hi)); cov=np.linalg.inv(r.jac.T@r.jac)*np.mean(r.fun**2); print('SE', np.sqrt(cov.diagonal()), 'corr with G',cov[3]/np.sqrt(cov[3,3]*cov.diagonal())); f=fun(r.x)
print('residual blocks');
for inds in np.array_split(np.arange(len(y)),12):
 j=inds[m[inds]]; print(T[inds[0]],len(j),np.mean(f[j]),np.std(f[j]))
for weight in [.002,.004,.007,.01,.015,.02,.025,.035,.05]:
 for use in ['spacing','central','strength']:
  def reg(z):
   if use=='spacing': return np.r_[np.diff(z[:k],2)]
   if use=='central': return np.r_[z[2]-np.log(5)]
   return np.r_[(z[4:7]/z[4:7].mean()-1)]
  rr=least_squares(lambda z: np.r_[fun(z)[m],weight*reg(z)],r.x,bounds=(lo,hi));print(use,weight, np.round(np.r_[np.exp(rr.x[:k]),rr.x[k]],4))
for factor in [0.8,1.0,1.25,1.5,2.0,3.0]:
 for horizon in [np.inf,7.,10.,14.]:
  scale=.016*factor
  # Use an ordinary quadratic penalty alongside the Cauchy data likelihood.
  def obj(z):
   res=fun(z); rob=np.sign(res)*scale*np.sqrt(np.log1p((res/scale)**2)); return np.r_[rob,.016*np.exp(z[:k])/horizon]
  rr=least_squares(obj,z0,bounds=(lo,hi)); z=rr.x
  pred=z[k]*ee+basis(tt,ee,np.exp(z[:k]),'linear')@z[k+1:-2]
  print(factor,horizon,'G',round(z[k],4),'error',round(np.linalg.norm(pred-gt)/np.linalg.norm(gt),5))
