import pickle,numpy as np
from itertools import combinations_with_replacement as cr
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
np.set_printoptions(precision=4,suppress=True,linewidth=150)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=52
x=np.r_[s['train']['x'],s['test']['x']]; tr=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
a=lambda z:np.c_[np.ones(len(z)),z]; q=lambda z:np.column_stack((a(z),*(np.prod(z[:,i],axis=1) for i in cr(range(z.shape[1]),2))))
basis=lambda r:-np.expm1(-t[:,None]*r)
r=least_squares(lambda r:(basis(r)@fit(basis(r),tr[:n].T)-tr[:n].T).ravel(),[1.8,.41]).x
B=np.linalg.qr(basis(r))[0]; Y=np.c_[y,tr@B]
u,v=s['panel_a'].values(); keep=np.arange(len(u))
for i in range(4):
 beta=fit(a(u[keep]),v[keep]); e=np.linalg.norm(a(u)@beta-v,axis=1);keep=np.argsort(e)[:38]
print('A',beta,'sorted residuals',np.sort(e)); print('A diagonal alternative', [fit(a(u[keep,j:j+1]),v[keep,j]) for j in range(4)])
bu,bv=s['panel_b'].values(); bb=fit(a(bu),bv); print('B',bb)
u,v1,v2=s['panel_c'].values(); maps={}
for j,v in enumerate([v1,v2],1):
 for name,fn in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('sin+cos',lambda w:np.c_[np.sin(w),np.cos(w)]),('rational',lambda w:1/(1+w*w))]:
  f=lambda z,r:np.c_[a(z),fn(z*r)]
  obj=lambda r:(f(u,r)@fit(f(u,r),v)-v).ravel()
  opt=min((least_squares(obj,[p]*4,bounds=(.15,6)) for p in [.8,1.5,2.5]),key=lambda o:np.linalg.norm(o.fun)); rr=opt.x; d=f(u,rr); co=fit(d,v)
  loo=np.mean(abs(opt.fun.reshape(-1,4)/(1-np.sum(np.linalg.qr(d)[0]**2,axis=1))[:,None])); print(j,name,rr,np.mean(abs(opt.fun)),loo)
  if (j,name) in [(1,'tanh'),(2,'cos'),(2,'sin+cos')]:
   print(co);maps[str(j)+name]=lambda z,rr=rr,fn=fn,co=co:np.c_[a(z),fn(z*rr)]@co
print('CV terms: y,traj')
z0=a(x)@beta; coords={'x':x,'A':z0,'Bx':a(x)@bb,'BA':a(z0)@bb}
for key,m in maps.items():
 for label,z in list(coords.items())[:4]:coords[key+label]=m(z)
def cv(p,ker=None):
 coef=fit(p[:n],Y[:n]) if ker is None else fit(np.linalg.cholesky(ker[:n,:n]+.003*np.eye(n)),np.c_[p[:n],Y[:n]])
 if ker is None: pred=p[n:72]@coef
 else:
  w=fit(coef[:,:p.shape[1]],coef[:,p.shape[1]:]);pred=p[n:72]@w+ker[n:72,:n]@np.linalg.solve(ker[:n,:n]+.003*np.eye(n),Y[:n]-p[:n]@w)
 return np.mean(abs(pred[:,0]-y[n:])),np.mean(abs(pred[:,1:]@B.T-tr[n:]))
k=np.exp(-.08*cdist(x,x,'sqeuclidean'))
for label,z in coords.items():print(label,cv(a(z)),cv(q(z)),cv(q(z),k));
for label in list(coords)[4:]:
 for label2 in ('x','A','Bx','BA'):
  z=coords[label];w=coords[label2]; pp=np.c_[q(w),z]; print(label,label2,cv(pp),cv(pp,k))
