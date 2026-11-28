from explore import *
from scipy.optimize import least_squares
u=pc['u']; v=pc['v1'];
for name,fn,start in [('tanh',np.tanh,2.3),('sin',np.sin,3.2),('rat',lambda a:a/(1+a*a),1.3)]:
 for shifts in [False,True]:
  def f(a,q): return np.c_[np.ones(len(q)),q,fn(q*a[:4]+(a[4:] if shifts else 0))]
  def fit(a,idx):
   F=f(a,u[idx]); b=np.linalg.lstsq(F,v[idx],rcond=None)[0]; return F@b-v[idx],b
  a0=np.r_[np.full(4,start),np.zeros(4)] if shifts else np.full(4,start); all=np.arange(len(u)); r=least_squares(lambda a:fit(a,all)[0].ravel(),a0,max_nfev=200); b=fit(r.x,all)[1]; errs=[]
  for i in all:
   ids=all[all!=i]; rr=least_squares(lambda a:fit(a,ids)[0].ravel(),a0,max_nfev=150); errs.append(f(rr.x,u[[i]])@fit(rr.x,ids)[1]-v[i]);
  print(name,shifts,'param',r.x.round(4),'rmse',np.mean(r.fun**2)**.5,'loo',np.mean(np.array(errs)**2)**.5,flush=True)
