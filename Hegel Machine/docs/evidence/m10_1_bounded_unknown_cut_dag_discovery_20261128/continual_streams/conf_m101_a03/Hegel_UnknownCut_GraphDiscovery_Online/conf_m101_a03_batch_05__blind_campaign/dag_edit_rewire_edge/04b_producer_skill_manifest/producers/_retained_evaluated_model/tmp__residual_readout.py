exec(open('/tmp/sparse_readout.py').read().split('for base in')[0])
from scipy.optimize import least_squares
u=x[:,1]-x[:,3]; p=z[:,0]*z[:,2]; q=z[:,1]*z[:,3]; B=np.c_[np.ones(n),z,p,q,np.sin(1.8*u)]
for k,v in enumerate(folds+[np.argsort(np.linalg.norm(x,axis=1))[-14:]]):
 t=np.setdiff1d(idx,v); A=B; R=F[t]-A[t]@np.linalg.lstsq(A[t],F[t],rcond=None)[0]; norm=np.linalg.norm(R,axis=0); R/=np.maximum(norm,1e-7); selected=[]; E=[]; r=y[t]-A[t]@np.linalg.lstsq(A[t],y[t],rcond=None)[0]; order=np.argsort(-np.abs(r@R)); print('residual',k,N[order[:9]])
 for j in range(5):
  jj=np.argmax(np.abs(r@R)); selected.append(jj); R[:,jj]=0; A=np.c_[B,F[:,selected]]; w=np.linalg.lstsq(A[t],y[t],rcond=None)[0]; r=y[t]-A[t]@w; E.append(np.mean(abs(A[v]@w-y[v])))
 print('greedy',N[selected],np.round(E,4),np.std(r))
# Test whether the apparent pair products and calibrated contrast are separate additive effects or modulate each other.
for kind in range(7):
 E=[]; par=[]
 for v in folds+[np.argsort(np.linalg.norm(x,axis=1))[-14:]]:
  t=np.setdiff1d(idx,v)
  def fit(a,ret=False):
   f,h,g=a; U=np.sin(f*(x[:,1]-h*x[:,3])); V=np.sin(g*p); Q=np.sin(g*q); Z=np.c_[np.ones(n),z]
   A=[np.c_[Z,p,q,U], np.c_[Z,V,Q,U], np.c_[Z,p,q,U,U*z.T.__array__().T] if False else np.c_[Z,p,q,U,U[:,None]*z], np.c_[Z,p,q,U,U*p,U*q],np.c_[Z,p,q,U,np.cos(f*(x[:,1]-h*x[:,3]))],np.c_[Z,p,q,U,z[:,0]/(1+g*x[:,1]**2)],np.c_[Z,p,q,U,p*p,q*q,p*q]][kind]; w=np.linalg.lstsq(A[t],y[t],rcond=None)[0]; r=A@w-y
   return (np.mean(abs(r[v])),np.std(r[t])) if ret else r[t]
  sols=[least_squares(fit,[f,1.,1.],bounds=([.3,.15,.01],[5.,3.,8.]),max_nfev=75) for f in [1.,2.,3.]]; m=min(sols,key=lambda m:m.cost); E.append(fit(m.x,True)); par.append(m.x)
 print('profile',kind,'errors',np.round(E,4).tolist(),'params',np.round(par,2).tolist())
