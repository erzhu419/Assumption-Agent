import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=150)
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def ones(a): return np.c_[np.ones(len(a)),a]
def poly(u,d):
 from itertools import combinations_with_replacement as cr
 return np.array([np.prod(u[:,j],axis=1) for k in range(1,d+1) for j in cr(range(4),k)]).T
p=s['panel_a']; u,v=p['u'],p['v']; X=ones(u); keep=np.ones(len(u),bool)
for i in range(5):
 c=ls(X[keep],v[keep]); err=np.linalg.norm(X@c-v,axis=1); keep=err<3*np.median(err)
print('A',c,'sing vals',np.linalg.svd(c[1:],compute_uv=False),'resid',np.sort(err));
p=s['panel_c']; u=p['u']; t=s['eval_times']; Y=s['train']['traj'];
for key in ['v1','v2']:
 v=p[key]; print(key)
 for k in [1,2,3]:
  ff=ones(poly(u,k)); pred=ff@ls(ff,v); print('poly',k,'rmse',np.mean((pred-v)**2)**.5)
 for name,fn in [('tanh',lambda u,k:np.tanh(k*u)),('aff+tanh',lambda u,k:.45*u+np.tanh(k*u)),('sin',lambda u,k:np.sin(k*u)),('sin+aff',lambda u,k:np.c_[u,np.sin(k*u)]),('cos',lambda u,k:np.cos(k*u)),('cos+aff',lambda u,k:np.c_[u,np.cos(k*u)]),('tanh+aff',lambda u,k:np.c_[u,np.tanh(k*u)]),('quad diag',lambda u,k:np.c_[u,u*u]),('sin+cos',lambda u,k:np.c_[np.sin(k*u),np.cos(k*u)]),('sin+cos+aff',lambda u,k:np.c_[u,np.sin(k*u),np.cos(k*u)])]:
  def loss(k):
   f=ones(fn(u,k)); return np.mean((f@ls(f,v)-v)**2)**.5
  grid=np.linspace(.2,12,121); ii=np.argmin([loss(k) for k in grid]); k=minimize_scalar(loss,bounds=(grid[max(ii-1,0)],grid[min(ii+1,len(grid)-1)]),method='bounded').x; print(name,round(k,5),round(loss(k),6))
for name,f in [('2 rise',lambda a,b:np.c_[-np.expm1(-a*t),-np.expm1(-b*t)]),('rise+pulse',lambda a,b:np.c_[-np.expm1(-a*t),t*np.exp(-b*t)]),('2 pulse',lambda a,b:np.c_[t*np.exp(-a*t),t*np.exp(-b*t)]),('rise+decay-rise',lambda a,b:np.c_[-np.expm1(-a*t),np.exp(-b*t)*(-np.expm1(-a*t))]),('linear+pulse',lambda a,b:np.c_[t,t*np.exp(-b*t)]),('sin+rise',lambda a,b:np.c_[np.sin(a*t),-np.expm1(-b*t)])]:
 def err(p):
  x=f(*p); return (x@ls(x,Y.T)-Y.T).ravel()
 z=min([least_squares(err,q,bounds=([.0001]*2,[12]*2)) for q in [[.3,1],[1,.3],[1,2],[2,1]]],key=lambda o:o.cost); print(name,z.x,np.mean(abs(z.fun)),np.max(abs(z.fun)))
