from panels import *
from itertools import combinations
from sklearn.model_selection import KFold
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize

splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=13).split(X)]
def normalized(x):
 x=x-x.mean(0);return x/np.maximum(x.std(0),1e-8)
def linear(x):
 f=normalized(x);return f@f.T/f.shape[1]
def radial(x,l):
 f=normalized(x);d=np.maximum(0,np.diag(f@f.T)[:,None]+np.diag(f@f.T)[None,:]-2*f@f.T);r=np.sqrt(5*d/f.shape[1])/l;return (1+r+r*r/3)*np.exp(-r)
def library(x,which):
 groups={}
 for n,d in routes(x):
  if n not in which:continue
  z,w=C1(d),C2(d)
  groups[n+' crossed C']=linear((z[:,:,None]*w[:,None,:]).reshape(len(d),-1))
  groups[n+' C quad']=linear(PolynomialFeatures(2,include_bias=False).fit_transform(np.c_[z,w]))
  for name,f in [('d',d),('C1',z),('C2',w),('Cmix',np.c_[z,w]),('diff',z-w),('prod',z*w)]:
   if name!='Cmix':
    h=f if name in ('diff','prod') else np.column_stack([f]+[f[:,i]+k*f[:,j] for i,j in combinations(range(4),2) for k in [-1,1]])
    groups[n+' '+name+' add']=sum(radial(h[:,i:i+1],1.0) for i in range(h.shape[1]))/h.shape[1]
   if name not in ('diff','prod'):
    for l in [.7,1.5]:groups[n+' '+name+str(l)]=radial(f,l)
  groups[n+' cubic']=linear(PolynomialFeatures(3,include_bias=False).fit_transform(d))
 groups['quadratic']=linear(PolynomialFeatures(2,include_bias=False).fit_transform(x))
 return groups

def predict(lib,y,i,j,shrink):
 k0=np.array(list(lib.values()));k0=k0-k0.mean(axis=1,keepdims=True)-k0.mean(axis=2,keepdims=True)+k0.mean(axis=(1,2),keepdims=True)
 k0/=np.maximum(np.diagonal(k0,axis1=1,axis2=2).mean(1),1e-6)[:,None,None]
 k0=np.concatenate((k0,np.ones_like(k0[:1]),np.eye(k0.shape[1])[None]),axis=0)
 k=k0[:,i[:,None],i];c=k0[:,j[:,None],i];m=len(k); yy=y[i]
 prior=np.r_[np.full(m-3,.15/(m-3)),.09,.12,.0006]; k=k.reshape(m,-1)
 def objective(w):
  a=np.exp(w); cs=cho_factor((a@k).reshape(len(i),len(i))); v=cho_solve(cs,yy); inv=cho_solve(cs,np.eye(len(i))); p=w-np.log(prior)
  return .5*yy@v+np.log(cs[0].diagonal()).sum()+.5*shrink*(p@p), .5*a*(k@(inv-np.outer(v,v)).ravel())+shrink*p
 r=minimize(objective,np.log(prior),jac=True,bounds=[(-16,2)]*m,method='L-BFGS-B',options={'maxiter':150});a=np.exp(r.x);v=cho_solve(cho_factor((a@k).reshape(len(i),len(i))),yy)
 return a@(c@v),a

if __name__=='__main__':
 print('A',ab,'B',B);print('A residual quantiles',np.quantile(abs(s['panel_a']['v']-s['panel_a']['u']*ab[:,1]-ab[:,0]),[.25,.5,.75,.9,.95,.99,1],axis=0))
 print('C derivatives',C1(np.r_[np.zeros((1,4)),np.eye(4)*.1]),C2(np.r_[np.zeros((1,4)),np.eye(4)*.1]))
 for ns in [('Ai','AiB','AiBi'),('X','XB','XBi'),('AiB',),('Ai',),('X','XB','Ai','AiB','XBi','AiBi')]:
  l=library(X,ns)
  for sh in [.025,.12,.5,2.]:
   err=[]
   for i,j in splits:
    h,a=predict(l,Y,i,j,sh);err.append(abs(h-Y[j]).mean())
    if len(err)==1:top=[(list(l)[h],round(a[h],4)) for h in np.argsort(-a[:-2])[:5]]
   print('adaptive',ns,sh,np.round(err,5),top,flush=True)
