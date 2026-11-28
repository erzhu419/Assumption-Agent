import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; one=lambda x:np.c_[np.ones(len(x)),x]
a=s['panel_a'];u=a['u'];v=a['v']; diag=[]
for j in range(4):
 i,k=np.triu_indices(len(u),1); d=u[i,j]-u[k,j]; q=np.abs(d)>1e-8; m=np.median((v[i[q],j]-v[k[q],j])/d[q]); dev=v[:,j]-m*u[:,j]; good=abs(dev-np.median(dev))<.005; diag.append(ls(one(u[good,j]),v[good,j]));
bias,scale=np.array(diag).T
b=s['panel_b']; B=ls(one(b['u']),b['v']); z=(s['train']['x']-bias)/scale; w=one(z)@B
tr=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; T=lambda p:-np.expm1(-t[:,None]*p); fit=least_squares(lambda p:(T(p)@ls(T(p),tr.T)-tr.T).ravel(),[2.,.5],bounds=([.2,.01],[10.,1.5])); h=T(fit.x); target=np.c_[y,ls(h,tr.T).T]; print('rates',fit.x,'time residual',np.mean(abs(h@target[:,1:].T-tr.T))); print('target std',target.std(0))
def powers(x,d):
 ix=[i for n in range(1,d+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),n)];return np.array([np.prod(x[:,i],axis=1) for i in ix]).T
for nm,x in [('raw',s['train']['x']),('z',z),('w',w)]:
 for deg in [1,2,3,4]:
  f=one(powers(x,deg)); c=ls(f[:52],target[:52]); r=f[52:]@c-target[52:]; print(nm,deg,np.round(np.mean(abs(r),0),5),np.mean(abs(r[:,1:]@h.T)))
 for fn in [np.sin,np.cos,np.tanh,lambda u:u/(1+abs(u)),lambda u:abs(u)]:
  for k in [.7,1.25,2,3]:
   f=one(np.c_[powers(x,2),fn(k*x)]);c=ls(f[:52],target[:52]);r=f[52:]@c-target[52:];print(nm,getattr(fn,'__name__',''),k,np.round(np.mean(abs(r),0),4),round(np.mean(abs(r[:,1:]@h.T)),4))
c=s['panel_c']; x=c['u'];
for part in ['v1','v2']:
 v=c[part]; print(part)
 for name,f in [('linear',x),('square',np.c_[x,x*x]),('fullquad',powers(x,2)),('abs',np.c_[x,abs(x)]),('xabs',np.c_[x,x*abs(x)]),('cube',np.c_[x,x*x,x**3])]:
  a=one(f); coeff=ls(a,v); rs=a@coeff-v; loo=rs/(1-np.diag(a@np.linalg.pinv(a)))[:,None]; print(name, np.std(rs),np.std(loo));
 for fn in [np.sin,np.cos,np.tanh,lambda u:u/(1+abs(u)), lambda u:np.maximum(u,0),lambda u:1/(1+u*u)]:
  f=lambda p:one(np.c_[x,fn(p[0]*x+p[1:])]); res=lambda p:(f(p)@ls(f(p),v)-v).ravel(); ans=[]
  for p in [[1,.1,.1,.1,.1],[1.,0,0,0,0],[2.,.7,.7,.7,.7],[3.,.7,-.7,.7,-.7]]:
   o=least_squares(res,p,max_nfev=400);ans.append((np.std(o.fun),o.x))
  print(getattr(fn,'__name__',''),min(ans,key=lambda v:v[0]));
