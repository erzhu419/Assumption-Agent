import pickle
from itertools import combinations_with_replacement
import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.optimize import least_squares, minimize

s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x,y,z=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; n=len(x)
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
def poly(a,d):
 return np.column_stack([np.ones(len(a))]+[np.prod(a[:,j],1) for k in range(1,d+1) for j in combinations_with_replacement(range(4),k)])
def kernel(a,b,par,family):
 dx=(a[:,None,:]-b[None,:,:]); r2=np.einsum('ijk,k,ijk->ij',dx,np.exp(par[:4]),dx)
 if family=='gauss': return np.exp(-r2)
 if family=='rq': return 1/(1+r2)
 if family=='mat52':
  r=np.sqrt(r2); return (1+r+r2/3)*np.exp(-r)
 if family=='mat72':
  r=np.sqrt(r2); return (1+r+0.4*r2+r2*r/15)*np.exp(-r)
 raise ValueError(family)
def predict(a,b,yy,family,degree,param=None):
 P=poly(a,degree); Q=poly(b,degree); m=P.shape[1]
 if param is None:
  N=qr(P,mode='full')[0][:,m:]; v=N.T@yy
  if v.ndim==2:
   u,d,_=np.linalg.svd(v,full_matrices=False); v=u[:,:2]*np.sqrt(np.sum(d[:2]**2)/2)
  else: v=v[:,None]
  def obj(p):
   K=N.T@kernel(a,a,p,family)@N+np.exp(p[-1])*np.eye(len(N.T))
   f=cho_factor(K); scale=np.sum(v*cho_solve(f,v))/v.size
   return np.log(scale)+2*np.log(np.diag(f[0])).mean()
  o=minimize(obj,np.log([.24]*4+[.002]),method='L-BFGS-B',bounds=[(-4,1)]*4+[(-12,-1)],options={'maxiter':70}); param=o.x
 K=kernel(a,a,param,family)+np.exp(param[-1])*np.eye(len(a))
 f=cho_factor(K); A=cho_solve(f,P); v=cho_solve(f,yy); c=np.linalg.solve(P.T@A,P.T@v)
 return Q@c+kernel(b,a,param,family)@(v-A@c),np.exp(param)
splits={'fixed':np.arange(52,72),'radial':np.argsort(np.linalg.norm(x,axis=1))[-18:]}
for j in range(4):
 order=np.argsort(x[:,j]); splits[f'low{j}']=order[:18]; splits[f'high{j}']=order[-18:]
for i,ix in enumerate(np.array_split(np.random.default_rng(712).permutation(n),4)): splits[f'rand{i}']=ix
scores={}; picks={}
for label,va in splits.items():
 tr=np.setdiff1d(np.arange(n),va); Z=z[tr].T
 def bas(k):return -np.expm1(-t[:,None]*k)
 b=bas(least_squares(lambda k:(bas(k)@ls(bas(k),Z)-Z).ravel(),[1.5,.25],bounds=([.7,.02],[4,.65])).x)
 # Learn the state geometry using two orthogonalized modes, not correlated amplitudes.
 U=np.linalg.qr(b)[0]; H=Z.T@U
 def add(key,p,q):
  e=np.abs(q-z[va]); scores.setdefault(key,[]).append([np.mean(abs(p-y[va])),e.mean(),e.mean(0).max()])
 def pred(f,d,py,pz): return predict(x[tr],x[va],y[tr],f,d,py),predict(x[tr],x[va],H,f,d,pz)
 (p,_),(q,_)=pred('gauss',2,np.log([.23]*4+[.001]),np.log([.23]*4+[.006])); q=q@U.T; add('baseline',p,q)
 for f in ['gauss','rq','mat52','mat72']:
  for d in [1,2]:
   (pp,wp),(qq,wq)=pred(f,d,None,None); qq=qq@U.T
   add((f,d),pp,qq); add((f,d,'blend'),.5*(pp+p),.5*(qq+q)); picks.setdefault((f,d),[]).append(np.r_[wp,wq])
 for f,h,reg in [('rq',.2,.001),('rq',.4,.001),('mat52',1.,.001),('mat72',1.5,.001)]:
  (pp,_),(qq,_)=pred(f,2,np.log([h]*4+[reg]),np.log([h]*4+[reg*4])); add((f,h),pp,qq@U.T)
for key,vals in scores.items():
 a=np.array(vals); print(key,'fixed',np.round(a[0],5),'tail max',np.round(a[1:10].max(0),5),[list(splits)[1+int(i)] for i in a[1:10].argmax(0)],'rand max',np.round(a[10:].max(0),5),'tail mean',np.round(a[1:10].mean(0),5))
 if key in picks: print('  median fitted',np.round(np.median(picks[key],axis=0),4))
