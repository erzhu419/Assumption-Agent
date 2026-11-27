import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/chamber_probe.npz'); p,h,x,dt,t,tau,th=[z[k] for k in ('past','hold','features','dt','t','taus','theta')]; n=len(h); e=z['er']; E=z['ec']; after=np.cumsum(dt[::-1])[::-1]-dt
mem=lambda a:p@(-np.expm1(-dt[:,None]/np.atleast_1d(a))*np.exp(-after[:,None]/np.atleast_1d(a)))
m=mem(tau); f=-np.expm1(-t[:,None]/tau); c=z['corr']; np.set_printoptions(precision=5,suppress=True)
def cv(A,Y):
 B=np.linalg.lstsq(A,Y,rcond=None)[0]; q=np.linalg.qr(A,mode='reduced')[0]; R=Y-A@B; den=1-(q*q).sum(1); O=R/den.reshape((-1,)+(1,)*(Y.ndim-1)); return B,np.sqrt(np.mean(R**2,axis=0)),np.sqrt(np.mean(O**2,axis=0)),np.max(abs(O),axis=0)
o=np.ones(n); H=np.c_[o,h,m]; print('memory',m[:5]); print('rate remainder',np.c_[e,e-th[0]/tau[0]*m[:,0],e-m@(th[:2]/tau)])
for name,A in [('fast',np.c_[o,m[:,0]]),('both',np.c_[o,m]),('mem and hold',H),('history',np.c_[o,h,p]),('fast plus feature',np.c_[o,m[:,0],x]),('history + feature',np.c_[o,h,p,x])]: print(name,cv(A,e))
for name,Y in [('fast',c[:,0]),('slow',c[:,1]),('drift',c[:,2]),('shape1',E[:,3]-e*f[3,0]*tau[0]),('shape2',E[:,-1]-e*f[-1,0]*tau[0])]:
 print(name)
 for nm,A in [('H',H),('H+x',np.c_[H,x]),('H+hold*x',np.c_[H,h[:,None]*x]),('H+mem1*x',np.c_[H,m[:,1,None]*x]),('H+gap1*x',np.c_[H,(h-m[:,1])[:,None]*x]),('H+gap0*x',np.c_[H,(h-m[:,0])[:,None]*x]),('H+prev*x',np.c_[H,p[:,-1,None]*x])]: print(nm,cv(A,Y))
print('rate single tau free')
for extra in [o[:,None],np.c_[o,h],np.c_[o,h,m[:,1]]]:
 def fun(a):
  A=np.c_[extra,mem(np.exp(a[0]))]; return A@np.linalg.lstsq(A,e,rcond=None)[0]-e
 fit=least_squares(fun,[np.log(tau[0])]); a=np.exp(fit.x[0]); print(a,cv(np.c_[extra,mem(a)],e))
