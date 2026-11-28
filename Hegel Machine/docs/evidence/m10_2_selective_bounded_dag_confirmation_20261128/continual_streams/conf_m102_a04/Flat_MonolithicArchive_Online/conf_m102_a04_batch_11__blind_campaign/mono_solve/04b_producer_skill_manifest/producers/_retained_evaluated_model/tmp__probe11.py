import itertools as it,pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def one(f):return np.c_[np.ones(len(f)),f]
def ls(f,y):return np.linalg.lstsq(f,y,rcond=1e-10)[0]
def report(f,y):
 h=f@np.linalg.pinv(f,rcond=1e-10); r=y-h@y;return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/(1-h.diagonal()).reshape((-1,)+(1,)*(y.ndim-1)))**2))
def fit(u,y,fn,bounds=(.01,4)):
 qs=np.linspace(*bounds,51); fun=lambda q:report(fn(u,q),y)[0]; i=np.argmin([fun(q) for q in qs]); q=minimize_scalar(fun,bounds=(qs[max(i-1,0)],qs[min(i+1,50)]),method='bounded',options={'xatol':1e-10}).x; return q,report(fn(u,q),y)
a=s['panel_a']; keep=np.linalg.norm(a['v'],axis=1)<5; u,v=a['u'][keep],a['v'][keep]; np.set_printoptions(precision=4,suppress=True)
for i in range(4):
 j=(i+2)%4; pair=u[:,[i,j]];print('A',i,'range',v[:,i].min(),v[:,i].max())
 ids=[ix for k in range(4) for ix in it.combinations_with_replacement(range(4),k)]; f=np.array([u[:,ix].prod(1) for ix in ids]).T; print([(ix,round(w,4)) for ix,w in zip(ids,ls(f,v[:,i])) if abs(w)>.012]);
 for name,g in [('sin',lambda w,q:np.sin(q*w)),('tanh',lambda w,q:np.tanh(q*w)),('softsign',lambda w,q:w/(1+q*abs(w))),('rational',lambda w,q:w/(1+q*w*w)),('sinh',lambda w,q:np.sinh(q*w))]:
  for ix in (0,1,2,3):
   print(name,ix,fit(pair,v[:,i],lambda w,q:one(np.c_[w,g([w[:,0],w[:,1],w[:,0]+w[:,1],w[:,0]-w[:,1]][ix],q),w.prod(1)])))
p=s['panel_c']; u=p['u'];
for key in ('v1','v2'):
 print('C',key); y=p[key]
 for name,g in [('sin',lambda w,q:np.sin(q*w)),('tanh',lambda w,q:np.tanh(q*w)),('softsign',lambda w,q:w/(1+q*abs(w))),('rational',lambda w,q:w/(1+q*w*w)),('cos',lambda w,q:np.cos(q*w)),('sqrat',lambda w,q:w*w/(1+q*w*w)),('exp',lambda w,q:np.exp(-q*w*w))]:
  for ix in range(3):
   fn=lambda w,q:one(np.c_[g(w,q), w if ix>=1 else np.empty((len(w),0)),w*w if ix>=2 else np.empty((len(w),0))]); q,err=fit(u,y,fn); print(name,ix,round(q,6),np.round(err,7))
