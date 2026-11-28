import pickle,numpy as np
from itertools import combinations,combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
one=lambda a:np.c_[np.ones(len(a)),a]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,k],axis=1) for j in range(1,d+1) for k in combinations_with_replacement(range(x.shape[1]),j)])
p=s['panel_c']; u=p['u']; a=one(u); pairs=list(combinations(range(4),2)); prod=np.column_stack([u[:,i]*u[:,j] for i,j in pairs]); diff=np.column_stack([u[:,i]+sg*u[:,j] for i,j in pairs for sg in (-1,1)])
for key in ('v1','v2'):
 v=p[key]; print(key,'quadratic coefficients\n',ls(poly(u,2),v))
 for name,base,vals in [('one',a,u),('pair',a,prod),('diff',a,diff),('quadratic',poly(u,2),u)]:
  for fname,f in [('lin',lambda q:q),('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('sinc',lambda q:np.sinc(q/np.pi)),('cube',lambda q:q**3)]:
   out=[]
   for j in range(4):
    yy=v[:,j]; errs=[]
    for k in range(vals.shape[1]):
     for sc in (.65,1.,1.5,2.):
      b=np.c_[base,f(sc*vals[:,k])]; c=ls(b[:18],yy[:18]); errs.append((np.std(b[:18]@c-yy[:18]),np.std(b[18:]@c-yy[18:]),k,sc))
    out.append(min(errs))
   print(name,fname,np.array(out).T)
 for fname,fun in [('sin',np.sin),('tanh',np.tanh),('square',lambda q:q*q),('cos',np.cos)]:
  out=[]; params=[]
  for j in range(4):
   yy=v[:,j]; b=a[:18]; proj=np.eye(18)-b@np.linalg.pinv(b); best=None
   for k in range(12):
    rng=np.random.default_rng(k); init=np.r_[0,np.eye(4)[k]] if k<4 else rng.normal(0,.8,5)
    def res(q):
     f=proj@fun(b@q[:5]); return f*q[5]-proj@yy[:18]
    o=least_squares(res,np.r_[init,.3],max_nfev=120,bounds=(-15,15))
    if best is None or np.linalg.norm(o.fun)<np.linalg.norm(best.fun):best=o
   q=best.x; pred=q[5]*fun(a@q[:5]); pred+=a@ls(b,yy[:18]-pred[:18]); out.append([np.std(best.fun),np.sqrt(np.mean((pred[18:]-yy[18:])**2))]); params.append(q)
  print('ridge',fname,np.array(out).T,np.round(params,3))
