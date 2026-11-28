from modes import *
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from itertools import combinations
pairs=list(combinations(range(4),2)); triples=list(combinations(range(4),3))

def kernels(a,b,h,mode):
 d=(a[:,None,:]-b[None,:,:])**2*np.exp(2*h[:4]); d=np.moveaxis(d,-1,0); dot=a@b.T; k=np.exp(h[4])*(1+dot)+np.exp(h[5])*(1+dot)**2
 def f(d):
  v=np.sqrt(5*d); return (1+v+v*v/3)*np.exp(-v)
 if mode==0: return k+np.exp(h[6])*f(d.sum(0))
 if mode==1: return k+np.exp(h[6])*f(d.sum(0))+sum(np.exp(w)*f(di) for w,di in zip(h[7:11],d))+sum(np.exp(w)*f(d[i]+d[j]) for w,(i,j) in zip(h[11:],pairs))
 if mode==2: return k+np.exp(h[6])*f(d.sum(0))+sum(np.exp(w)*f(d[list(ij)].sum(0)) for w,ij in zip(h[7:],triples))+sum(np.exp(w)*f(di) for w,di in zip(h[11:],d))
 if mode==3:
  v=np.sqrt(3*d.sum(0)); return k+np.exp(h[6])*(1+v)*np.exp(-v)+sum(np.exp(w)*f(d[i]+d[j]) for w,(i,j) in zip(h[7:],pairs))

if __name__=='__main__':
 for name in ['ai','x','aib']:
  X=base[name]; print('COORD',name,flush=True)
  for j in range(3):
   yy=Z[:,j]; mu=yy[:n].mean(); sd=yy[:n].std(); Y=(yy[:n]-mu)/sd; a=X[:n]; b=X[n:]
   for mode,nh in enumerate([7,17,15,13]):
    def loss(h,ret=False):
     K=kernels(a,a,h,mode); K.flat[::n+1]+=np.exp(h[-1]); cf=cho_factor(K); w=cho_solve(cf,Y)
     if ret:return mu+sd*kernels(b,a,h,mode)@w
     return .5*(Y@w)+np.log(np.diag(cf[0])).sum()
    h=np.r_[np.full(4,-.7),-1.,-3.,-.5,np.full(nh-7,-3.),-8.]; fit=minimize(loss,h,method='L-BFGS-B',bounds=[(-5,3)]*4+[(-12,5)]*(nh-4)+[(-14,-1)],options={'maxiter':120}); pred=loss(fit.x,True); print(j,mode,round(fit.fun,2),round(np.mean(abs(pred-yy[n:])),5),np.round(fit.x,2),flush=True)
