from scalar_probe import *
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize,least_squares
from scipy.spatial.distance import cdist

def gp(p,state,y,what):
 a,b=p['scores'],state@p['modes'].T
 sy,qy=p['sy'],p['qy']; st,qt=p['st'][:,:4],p['qt'][:,:4]
 K,Kq=p['K'],p['Kq']; U,L,_=np.linalg.svd(K-K.mean(0)-K.mean(1)[:,None]+K.mean()); h=U[:,:4]*L[:4]**.5; hq=(Kq-K.mean(0)-Kq.mean(1)[:,None]+K.mean())@(U[:,:4]/L[:4]**.5)
 coords=[(sy,qy),(st,qt),(h,hq),(a,b)]; D=[]; Q=[]
 for x,z in coords:
  x,z=(x-x.mean(0))/x.std(0),(z-x.mean(0))/x.std(0); D.append((x[:,None,:]-x)**2); Q.append((z[:,None,:]-x)**2)
 m=len(y); A=np.eye(m); basis=[K,np.ones_like(K),a@a.T]; query=[Kq,np.ones_like(Kq),b@a.T]
 def mat(w,Ds,B):
  z=np.exp(w); mats=[r*z[i] for i,r in enumerate(B)]; jac=list(mats)
  j=len(B)
  for i in what:
   d=Ds[i]; k=d.shape[-1]; q=d*z[j+1:j+1+k]; f=z[j]*np.exp(-q.sum(-1)); mats.append(f); jac.extend([f,*np.moveaxis(-f[:,:,None]*q,-1,0)]); j+=k+1
  return sum(mats),jac
 def obj(w):
  H,jac=mat(w,D,basis); N=np.exp(w[-1])*A; C=cho_factor(H+N); c=cho_solve(C,y); I=cho_solve(C,A); E=I-np.outer(c,c); val=.5*y@c+np.log(np.diag(C[0])).sum(); grad=.5*np.einsum('ij,kij->k',E,np.array([*jac,N])); return val,grad
 init=[0,0,0]; bounds=[(-20,5)]*3
 for i in what:
  init.extend([0]+[-2]*D[i].shape[-1]);bounds.extend([(-20,6)]+[(-12,3)]*D[i].shape[-1])
 init.append(-9);bounds.append((-15,-3)); w=minimize(obj,init,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':180}).x; H,_=mat(w,D,basis); C=cho_factor(H+np.exp(w[-1])*A); c=cho_solve(C,y); pred=mat(w,Q,query)[0]@c; loo=np.mean(np.abs(c/np.diag(cho_solve(C,A)))); return pred,loo,obj(w)[0]

def profile(f,g,a,b,y,kind):
 rng=np.random.default_rng(42); n,m=f.shape; sd=f.std(0); mu=f.mean(0); f,g=(f-mu)/sd,(g-mu)/sd; F=np.c_[np.ones(n),f,a]; G=np.c_[np.ones(len(g)),g,b]; R=np.linalg.qr(F,mode='reduced')[0]; ry=y-R@(R.T@y)
 def funcs(u):
  if kind=='sin':return np.c_[np.sin(u),np.cos(u)]
  if kind=='tanh':return np.tanh(u)[:,None]
  if kind=='rat':return (1/(1+u*u))[:,None]
 def residual(v):
  h=funcs(f@v[:m]+v[-1]); h-=R@(R.T@h); return ry-h@np.linalg.lstsq(h,ry,rcond=1e-8)[0]
 best=None
 for i in range(10):
  ini=rng.normal(0,.55,m+1); o=least_squares(residual,ini,max_nfev=100,bounds=(-5,5)); val=np.linalg.norm(o.fun)
  if best is None or val<best[0]:best=(val,o.x)
 val,v=best; w=np.linalg.lstsq(np.c_[F,funcs(f@v[:m]+v[-1])],y,rcond=1e-8)[0]; return np.c_[G,funcs(g@v[:m]+v[-1])]@w,val/np.sqrt(n),np.round(v,2)
if __name__=='__main__':
 for tr,va in [(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(np.arange(72)[np.arange(72)%3!=1],np.arange(72)[np.arange(72)%3==1])]:
  p,state,y,q,t,x,xq=fold(tr,va); a,b=p['scores'],state@p['modes'].T; print('FOLD',va[:4]);
  for what in [(0,),(1,),(2,),(3,),(0,1,2,3),(1,2),(1,3),(2,3),()]:
   pred,loo,obj=gp(p,state,y,what); print('gp',what,round(np.abs(q-pred).mean(),5),round(loo,5),round(obj,3))
  for label,f,g in [('x',x,xq),('sy',p['sy'],p['qy']),('st',p['st'][:,:4],p['qt'][:,:4]),('x+a',np.c_[x,a],np.c_[xq,b])]:
   for kind in ['sin','tanh','rat']:
    pred,err,v=profile(f,g,a,b,y,kind); print('profile',label,kind,round(np.abs(q-pred).mean(),5),round(err,5),v)
