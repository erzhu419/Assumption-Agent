import pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar,least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def fit(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def one(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),k)])
p=S['panel_a'];x=p['u'];y=p['v'];
for X in [one(x),one(np.c_[x,x*x]),one(np.c_[x,x**3]),one(np.c_[x,x*x,x**3]),poly(x,2)]:
 w=np.ones(len(x));
 for _ in range(45):
  b=fit(X*w[:,None],y*w[:,None]); e=np.linalg.norm(X@b-y,axis=1);w=1/np.sqrt(np.maximum(e,.00015))
 good=e<max(.002,2*np.median(e));b=fit(X[good],y[good]); print('A',X.shape[1],sum(good),np.sort(np.linalg.norm(X@b-y,axis=1))[::4]); print(b.T)
p=S['panel_c'];x=p['u'];print('C1 quad',fit(poly(x,2),p['v1']).T);print('C2 Y',p['v2'][:8]);
for fun in [np.tanh,np.sin,lambda z:z/(1+z*z),lambda z:z/(1+abs(z))]:
 for i,v in enumerate(p['v2'].T):
  f=lambda b:b[0]+b[1]*fun(b[2]+x@b[3:])-v
  r=min([least_squares(f,np.r_[0,1,0,np.random.default_rng(k).normal(size=4)],max_nfev=150) for k in range(10)],key=lambda r:r.cost); print('C2',fun.__name__,i,np.sqrt(np.mean(r.fun**2)),r.x)
h=S['train']['traj'];t=S['eval_times']; y=S['train']['y'];print('y vs h corr',np.corrcoef(np.c_[y,h].T)[0]);
for fun in [lambda t,k: -np.expm1(-k*t),lambda t,k:np.log1p(k*t),lambda t,k:t/(1+k*t),lambda t,k:t**k,np.sin]:
 def errs(a):
  v=fun(t,a); return np.linalg.norm(h-(h@v/np.sum(v*v))[:,None]*v,axis=1)
 r=minimize_scalar(lambda a:np.mean(errs(a)**2),bounds=(.01,6),method='bounded'); print('time',r.x,np.mean(errs(r.x)**2)**.5)
