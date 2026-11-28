import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=8,suppress=True)
def aff(u):return np.c_[np.ones(len(u)),u]
def ls(u,v):return np.linalg.lstsq(u,v,rcond=None)[0]
u,v=[s['panel_a'][k] for k in ('u','v')]
y=v-.18*np.tanh(1.8*u)+.32*np.sin(2.4*u[:,[2,3,0,1]])
m=np.ones_like(y,bool)
for _ in range(5):
 w=np.array([ls(aff(u[m[:,j],j]),y[m[:,j],j]) for j in range(4)]).T
 e=y-w[0]-w[1]*u;m=np.abs(e)<.5
print('sensor coefficients',w,'inlier count',m.sum(0),'resid quantiles',np.quantile(abs(e[m]),[0,.5,.9,1]),'outliers',np.sort(abs(e[~m])))
u,v=[s['panel_b'][k] for k in ('u','v')]
for kind,dim in [('shared',2),('phases',5),('individual',8)]:
 def des(p,u):
  k,ph=(p[0],p[1]) if dim==2 else ((p[0],p[1:]) if dim==5 else (p[:4],p[4:]))
  return aff(np.cos(k*u+ph))
 def residual(p,us=u,vs=v):
  A=des(p,us);return (A@ls(A,vs)-vs).ravel()
 best=[]
 for freq in [.5,1.,1.6,2.2,2.8,3.5,4.1]:
  for phase in [-.35,0.,.35]:
   start=np.r_[np.full(4 if dim==8 else 1,freq),np.full(1 if dim==2 else 4,phase)]
   best.append(least_squares(residual,start,max_nfev=150))
 opt=min(best,key=lambda x:x.cost)
 print(kind,'p',opt.x,'resid',np.linalg.norm(opt.fun),'max',abs(opt.fun).max(),'weights',ls(des(opt.x,u),v))
