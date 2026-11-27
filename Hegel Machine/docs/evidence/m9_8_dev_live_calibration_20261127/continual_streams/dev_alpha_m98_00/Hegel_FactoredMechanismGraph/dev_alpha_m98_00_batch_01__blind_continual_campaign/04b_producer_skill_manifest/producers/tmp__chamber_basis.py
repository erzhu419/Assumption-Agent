from rheology_base import *
from scipy.optimize import least_squares
Y=np.column_stack((q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs'])); past,hold,xx=pq
w=np.r_[np.ones(len(t)),.75]
def basis(tt,degree):
 return np.row_stack((np.column_stack((-np.expm1(-t[:,None]/tt),*[t**j for j in range(1,degree+1)])),np.r_[1/tt,1,np.zeros(degree-1)]))
def project(tt,degree):
 B=basis(tt,degree); coef=np.linalg.lstsq(B*w[:,None],(Y*w).T,rcond=None)[0].T
 return coef,(coef@B.T-Y)*w
for deg in [1,2,3]:
 for fixed in [True,False]:
  tt=tau if fixed else np.exp(least_squares(lambda p:project(np.exp(p),deg)[1].ravel(),np.log(tau),bounds=(np.log(.1),np.log(25)),max_nfev=120).x)
  C,E=project(tt,deg); print('degree',deg,'fixed',fixed,'tau',tt,'ratio',tt/tau,'curve/rate',np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean((E[:,-1]/w[-1])**2))); print('quartiles',np.round(np.percentile(C,[0,25,50,75,100],axis=0),4))
  if deg!=1:continue
  for j in range(2):
   print('branch',j)
   candidates=[]
   for name,order,dorder in [('normal',np.arange(len(dt)),dt),('reverse both',np.arange(len(dt))[::-1],dt[::-1]),('reverse stress',np.arange(len(dt))[::-1],dt),('reverse dt',np.arange(len(dt)),dt[::-1])]:
    for tf in [tau[j],tt[j]]:
     z=np.zeros(len(hold))
     for k,step in zip(order,dorder):z+=(-np.expm1(-step/tf))*(past[:,k]-z)
     F=np.column_stack((np.ones(len(hold)),hold,z)); coef=np.linalg.lstsq(F,C[:,j],rcond=None)[0]
     candidates.append((np.std(F@coef-C[:,j]),name,tf,np.round(coef,4)))
   print(sorted(candidates,key=lambda x:x[0])[:5]); F=np.column_stack((np.ones(len(hold)),hold,past)); fit=np.linalg.lstsq(F,C[:,j],rcond=None)[0]; print('unrestricted linear stress',np.round(fit,4),'sd',np.std(F@fit-C[:,j])); print('normalized amplitudes',np.round(C[:,j]/b[j],3))
  drift=C[:,2]; print('drift residual',np.round(drift-xx@b[2:],4)); print('drift residual corr [past,hold,X,drift,branch]',np.round(np.corrcoef(np.column_stack((drift-xx@b[2:],past,hold,xx[:,1:],xx@b[2:],C[:,:2])).T)[0,1:],3))
