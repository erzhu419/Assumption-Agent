from graph_probe import *
from scipy.linalg import cholesky,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(q,t,r,mode,trend,kind,details=False):
 L=np.c_[np.ones(len(q)),q]; D=poly(q,trend); n,p=q.shape; iu=np.tril_indices(p,-1)
 delta=q[:,None,:]-q[None,:,:]; outer=delta[:,:,:,None]*delta[:,:,None,:]; eye=np.eye(n)
 def metric(a):
  E=np.diag(np.exp(a[:p]));
  if mode=='full':E[iu]=a[p:-2]
  return E
 def cov(d,a):
  if kind=='rbf':return np.exp(-d/2),-.5*np.exp(-d/2)
  d=np.sqrt(np.maximum(5*d,1e-24)); return (1+d+d*d/3)*np.exp(-d),-5/6*(1+d)*np.exp(-d)
 def calc(a):
  E=metric(a); d=np.einsum('abij,ij->ab',outer,E@E.T); C,g=cov(d,a); R=np.exp(a[-1]); V=np.exp(a[-2]); Q=V*C+R*eye; F=cholesky(Q,lower=True); w=cho_solve((F,True),np.c_[D,t]); W=w[:,:D.shape[1]]; T=D.T@W; J=cholesky(T,lower=True); b=cho_solve((J,True),D.T@w[:,-1]); al=w[:,-1]-W@b; G=cho_solve((F,True),eye)-W@cho_solve((J,True),W.T)-np.outer(al,al); H=np.einsum('ab,abij->ij',G*g*V,outer)@E; H0=np.diag(H)*np.exp(a[:p]); grad=np.r_[H0,H[iu] if mode=='full' else [],.5*V*np.sum(G*C),.5*R*np.trace(G)]; return np.log(F.diagonal()).sum()+np.log(J.diagonal()).sum()+.5*(t-D@b)@al,grad,b,al
 a0=np.r_[np.full(p,-.6),np.zeros(len(iu[0]) if mode=='full' else 0),-2.5,-8.8]; bd=[(-3,2)]*p+[(-3,3)]*(len(a0)-p-2)+[(-10,2),(-13,-4)]; opt=minimize(lambda a:calc(a)[:2],a0,jac=True,method='L-BFGS-B',bounds=bd,options={'maxiter':120}); a=opt.x; _,_,b,al=calc(a); E=metric(a); d=cdist(r@E,q@E,'sqeuclidean'); pred=poly(r,trend)@b+np.exp(a[-2])*cov(d,a)[0]@al
 return (pred,a) if details else pred

if __name__=='__main__':
 for key in ('panel_a','panel_b','panel_c'):
  p=s[key]; d=poly(p['u'],1)
  for k in p:
   if k=='u':continue
   f=affine(p,k); e=np.linalg.norm(p[k]-d@f,axis=1); print(key,k,'row residual quantiles',np.quantile(e,[0,.5,.8,.9,1]),'robust affine',f.round(5))
 for j in range(2):
  for name in ('x','a-','a-b+'):
   q=frames(x)[name]; t=Y[:,j];
   for mode in ('diag','full'):
    for trend in (1,2):
     for kind in ('rbf','m52'):
      pp=[fit(q[a],t[a],q[b],mode,trend,kind,True) for a,b in splits]; print(j,name,mode,trend,kind,np.round([np.mean(abs(pred-t[b])) for (a,b),(pred,_) in zip(splits,pp)],5),'fit',np.round(pp[0][1],2),flush=True)
