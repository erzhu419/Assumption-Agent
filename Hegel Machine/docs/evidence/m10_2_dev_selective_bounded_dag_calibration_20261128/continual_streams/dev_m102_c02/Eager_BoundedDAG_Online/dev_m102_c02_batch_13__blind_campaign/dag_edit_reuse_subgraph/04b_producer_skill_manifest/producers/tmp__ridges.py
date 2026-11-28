import numpy as np
from scipy.optimize import least_squares
D=np.load('/tmp/readprobe.npz'); x=D['c']; y=D['y'];
fs={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'gauss':lambda z:np.exp(-z*z),'softabs':lambda z:np.sqrt(1+z*z),'rational':lambda z:z/(1+z*z),'rat2':lambda z:1/(1+z*z),'cosh':lambda z:np.cosh(np.clip(z,-20,20)), 'exp':lambda z:np.exp(np.clip(z,-20,20)), 'cubic':lambda z:z**3,'sigquad':lambda z:z*z/(1+z*z)}
def fit(x,y,x2,name,ns=1,extra=False,starts=22):
 rng=np.random.default_rng(6); f=fs[name]; a=np.column_stack((np.ones(len(x)),x)); a2=np.column_stack((np.ones(len(x2)),x2));
 if extra:
  j,k=np.triu_indices(4); a=np.column_stack((a,x[:,j]*x[:,k])); a2=np.column_stack((a2,x2[:,j]*x2[:,k]));
 Q=np.linalg.qr(a)[0]; r=y-Q@(Q.T@y)
 def res(p):
  h=f(a[:,:5]@p.reshape(ns,5).T); h-=Q@(Q.T@h); b=np.linalg.lstsq(h,r,rcond=1e-10)[0]; return h@b-r
 best=None
 for i in range(starts):
  p=rng.normal(0,[.35,.8,1.6][i%3],(ns,5)); p[:,0]*=.8
  opt=least_squares(res,p.ravel(),max_nfev=160,ftol=1e-8)
  if best is None or opt.cost<best.cost:best=opt
  if np.mean(abs(best.fun))<.005:break
 p=best.x.reshape(ns,5); b=np.linalg.lstsq(np.column_stack((a,f(a[:,:5]@p.T))),y,rcond=1e-10)[0]; out=np.column_stack((a2,f(a2[:,:5]@p.T)))@b
 return out,np.mean(abs(best.fun)),p,b
if __name__=='__main__':
 for n in fs:
  for ns,extra in [(1,False),(1,True),(2,False)]:
   h,tr,p,b=fit(x[:52],y[:52],x[52:],n,ns,extra,starts=12); te=np.mean(abs(h-y[52:])); print(n,ns,extra,'tr',round(tr,5),'te',round(te,5),'p',p.round(3),'b',b.round(3) if te<.03 else '',flush=True)
