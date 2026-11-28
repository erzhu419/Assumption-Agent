import numpy as np,pickle
from itertools import combinations
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; u,v=(s['panel_a'][k] for k in ('u','v')); h=np.c_[np.ones(len(u)),u]; ids=np.argsort(np.linalg.norm(h@np.linalg.lstsq(h,v,rcond=None)[0]-v,axis=1))[:-3]; u,v=u[ids],v[ids]; ij=np.array(list(combinations(range(4),2))); ijk=np.array(list(combinations(range(4),3)))
def af(x):return np.c_[np.ones(len(x)),x,np.prod(x[:,ij],axis=2),np.prod(x[:,ijk],axis=2)]
W=np.linalg.lstsq(af(u),v,rcond=None)[0];print('A squarefree coefficients, rmse',W.round(5),np.std(af(u)@W-v));
for i in range(4):
 j=(i+1)%4;k=(i+2)%4;l=(i+3)%4
 for name,F in [('cyclic',lambda x:np.c_[np.ones(len(x)),x[:,i],x[:,j],x[:,i]*x[:,j],x[:,i]*x[:,j]*x[:,k],x[:,i]*x[:,j]*x[:,l],x[:,i]*x[:,k]*x[:,l],x[:,j]*x[:,k]*x[:,l]]),('all',af)]:
  d=F(u);w=np.linalg.lstsq(d,v[:,i],rcond=None)[0]; print(i,name,np.std(d@w-v[:,i]),w.round(5))
def inverse(x,W=W):
 z=(x-W[0])@np.linalg.inv(W[1:5]);
 for _ in range(15):
  pred=af(z)@W; eps=1e-5; J=np.stack([(af(z+e*eps)@W-pred)/eps for e in np.eye(4)],axis=2); dz=np.linalg.solve(J,(pred-x)[...,None])[...,0]; z-=np.clip(dz,-.5,.5)
 return z
x=tr['x']; z=inverse(x); print('inverse',np.max(abs(af(z)@W-x)),np.quantile(z,[0,.05,.5,.95,1],axis=0));
def panel(k,name,f):
 p=s[k];u=p['u'];v=p[name]; d=lambda x,a:np.c_[np.ones(len(x)),x,f(x,a)] if f else np.c_[np.ones(len(x)),x]; err=lambda a:np.std(d(u,a)@np.linalg.lstsq(d(u,a),v,rcond=None)[0]-v); a=minimize_scalar(err,bounds=(.05,2),method='bounded').x if f else 0.; w=np.linalg.lstsq(d(u,a),v,rcond=None)[0]; return d(z,a)@w
b=panel('panel_b','v',None); c=panel('panel_c','v1',lambda x,k:x/(1+k*abs(x))); d=panel('panel_c','v2',lambda x,k:1/(1+k*x*x)); y=np.c_[tr['y'],tr['traj']];
def gram(a,kind):
 if kind=='poly': return np.c_[a,np.prod(a[:,ij],axis=2),a*a,np.prod(a[:,ijk],axis=2)]
 if kind=='rich':return np.c_[gram(a,'poly'),a**3,np.tanh(a),np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a),a/(1+abs(a)),1/(1+a*a),np.tanh(a[:,ij[:,0]]+a[:,ij[:,1]]),np.tanh(a[:,ij[:,0]]-a[:,ij[:,1]])]
 return np.c_[gram(a,'poly'),gram(np.tanh(a),'poly')]
if __name__=='__main__':
 for name,a in [('x',x),('z',z),('b',b),('c',c),('d',d),('cz',c+z),('dz',d+z)]:
  for kind in ['poly','tanh','rich']:
   A=gram(a,kind); errs=[]
   for reg in [.0001,.003,.03,.3,1,3]:
    ls=[]
    for T,V in [(np.arange(52),np.arange(52,72))]+[(np.delete(np.arange(52),vi),vi) for vi in np.array_split(np.random.default_rng(132).permutation(52),4)]:
     m=A[T].mean(0); st=np.maximum(A[T].std(0),1e-9); F=(A[T]-m)/st; P=(A[V]-m)/st@F.T@np.linalg.solve(F@F.T+reg*np.eye(len(T)),y[T]-y[T].mean(0))+y[T].mean(0); E=abs(P-y[V]);ls.append([E[:,0].mean(),E[:,1:].mean()]);
    errs.append((reg,np.round(ls[0],4),np.round(np.mean(ls[1:],axis=0),4)))
   print(name,kind,errs)
