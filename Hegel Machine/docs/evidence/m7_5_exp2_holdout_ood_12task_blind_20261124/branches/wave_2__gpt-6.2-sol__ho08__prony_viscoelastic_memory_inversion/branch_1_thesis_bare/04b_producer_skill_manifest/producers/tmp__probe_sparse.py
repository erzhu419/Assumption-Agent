from probe_prony import *
from scipy.optimize import least_squares

def matrix(t,x,l):
 tau=np.exp(l); z=np.diff(t)[:,None]/tau; a=np.exp(-z); f=-np.expm1(-z)/z; da=z*a; df=f-a
 h=np.zeros((len(t),len(tau))); dh=h.copy()
 for i,d in enumerate(np.diff(x)):
  dh[i+1]=a[i]*dh[i]+da[i]*h[i]+df[i]*d
  h[i+1]=a[i]*h[i]+f[i]*d
 return np.column_stack((x,h)),dh

def sparse(e):
 t=e['train_t']; x=e['train_strain']; y=e['train_stress_raw']; D=np.column_stack((np.ones(len(t)),(t-t.mean())/np.ptp(t)))
 grid=np.geomspace(.02,20,41); A=design(t,x,grid); w=np.ones(len(t))
 for _ in range(15):
  b,d,r=fit(A,D,y,w); sc=max(1.4826*np.median(np.abs(r-np.median(r))),1e-12); w=np.minimum(1,1.345*sc/np.maximum(np.abs(r),1e-15))
 good=np.abs(r)<5*sc; w=good.astype(float); b,d,r=fit(A,D,y,w)
 ans=[]
 for m in range(1,6):
  starts=[np.geomspace(.12,11,m) if m>1 else [2],np.geomspace(.35,7,m) if m>1 else [.5]]
  best=None
  for start in starts:
   b0,d0,_=fit(design(t,x,np.asarray(start)),D,y,w)
   def fun(p):
    X,_=matrix(t,x,p[:m]); return (X@p[m:-2]+D@p[-2:]-y)[good]
   def jac(p):
    X,dh=matrix(t,x,p[:m]); return np.column_stack((dh*p[m+1:-2],X,D))[good]
   res=least_squares(fun,np.r_[np.log(start),b0,d0],jac=jac,bounds=(np.r_[np.full(m,np.log(.02)),np.zeros(m+1),-np.inf,-np.inf],np.r_[np.full(m,np.log(20)),np.full(m+3,np.inf)]),x_scale='jac',max_nfev=110)
   if best is None or res.cost<best.cost: best=res
  q=best.x; rss=2*best.cost; n=good.sum(); bic=n*np.log(rss/n)+(2*m+3)*np.log(n)
  pred=design(e['test_t'],e['test_strain'],np.exp(q[:m]))@q[m:-2]
  ans.append((m,q[m],bic,np.exp(q[:m]),q[m+1:-2],pred,rss))
 return ans
if __name__=='__main__':
 for m,g,bic,tau,coef,pred,rss in sparse(e): print(m,g,bic,tau,coef,np.linalg.norm(pred-p['gt_test_stress'])/np.linalg.norm(p['gt_test_stress']),rss)
