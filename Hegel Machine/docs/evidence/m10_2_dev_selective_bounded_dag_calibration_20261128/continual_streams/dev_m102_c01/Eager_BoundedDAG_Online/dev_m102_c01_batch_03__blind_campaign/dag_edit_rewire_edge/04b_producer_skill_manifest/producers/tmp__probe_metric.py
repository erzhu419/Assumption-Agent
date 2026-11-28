from scalar_probe import *
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve

def fit(x,z,a,b,y,kind=0):
 mu=x.mean(0); sd=x.std(0); x,z=(x-mu)/sd,(z-mu)/sd; n,d=x.shape; I=np.eye(n); dx=x[:,None]-x; dz=z[:,None]-x; aa=a@a.T; ba=b@a.T
 i,j=np.tril_indices(d); diag=i==j
 def cov(p,u,v,D,S,train=False):
  l=np.zeros((d,d)); l[i,j]=np.where(diag,np.exp(p[:len(i)]),p[:len(i)]); vpar=np.exp(p[len(i):]); T=D@l; d2=(T*T).sum(-1); K=vpar[0]*np.exp(-d2); G=u@v.T/d; L=vpar[1]*(1+G)**2; C=vpar[2]*S; H=K+L+C+vpar[3]
  if not train:return H
  jac=np.empty((len(p),n,n)); jac[:len(i)]=(-2*K[:,:,None]*D[:,:,i]*T[:,:,j]*np.where(diag,l[i,j],1)).transpose(2,0,1); jac[len(i):]=[K,L,C,np.full_like(K,vpar[3]),vpar[4]*I]; return H+vpar[4]*I,jac
 def loss(p):
  H,J=cov(p,x,x,dx,aa,True); F=cho_factor(H); al=cho_solve(F,y); E=cho_solve(F,I)-np.outer(al,al); penalty=.05*np.sum(p[:len(i)]**2); reg=np.r_[.1*p[:len(i)],np.zeros(5)]; return .5*y@al+np.log(np.diag(F[0])).sum()+penalty,.5*np.einsum('ij,kij->k',E,J)+reg
 rng=np.random.default_rng(0); result=[]
 for k in range(6):
  ini=np.r_[np.where(diag,-1.5,0)+rng.normal(0,.25*k/5,len(i)),0,-3,-1,-1,-9]; opt=minimize(loss,ini,jac=True,method='L-BFGS-B',bounds=[(-5,1) if t else (-3,3) for t in diag]+[(-16,5)]*4+[(-14,-3)],options={'maxiter':180}); result.append(opt)
 p=min(result,key=lambda o:o.fun).x; H,_=cov(p,x,x,dx,aa,True); F=cho_factor(H); c=cho_solve(F,y); return cov(p,z,x,dz,ba)@c,p,loss(p)[0]
if __name__=='__main__':
 for va in [np.arange(52,72),np.arange(20),*[np.arange(k,72,3) for k in range(3)]]:
  tr=np.setdiff1d(np.arange(72),va); p,state,y,v,t,x,q=fold(tr,va); a,b=p['scores'],state@p['modes'].T; print('FOLD',va[:3],flush=True)
  for key,u,z in [('x',x,q),('st',p['st'][:,:4],p['qt'][:,:4]),('sy',p['sy'],p['qy']),('x+a',np.c_[x,a],np.c_[q,b])]:
   yh,w,obj=fit(u,z,a,b,y); print(key,round(np.mean(abs(v-yh)),6),round(obj,2),np.round(w,2),flush=True)
