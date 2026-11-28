import numpy as np,pickle
from scipy.optimize import minimize_scalar, least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def fns():
 return dict(abs=lambda x,k: x/(1+k*abs(x)),rats=lambda x,k:x/(1+k*x*x),ratq=lambda x,k:x*x/(1+k*x*x),ratc=lambda x,k:x**3/(1+k*x*x),tanh=lambda x,k:np.tanh(k*x),sin=lambda x,k:np.sin(k*x),cos=lambda x,k:np.cos(k*x),soft=lambda x,k:x/np.sqrt(1+k*x*x),log=lambda x,k:np.log1p(k*x*x),exp=lambda x,k:np.exp(-k*x*x),powers=lambda x,k:np.c_[x*x,x**3],cross=lambda x,k:np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]))
for tag in 'abc':
 p=s['panel_'+tag]; x=p['u']; print(tag,'input range',x.min(0),x.max(0))
 for key in p.keys()-{'u'}:
  y=p[key]; a=aff(x); e=np.linalg.norm(y-a@ls(a,y),axis=1); m=e<=4*np.median(e); print(key,'affine med',np.median(e),'trim rms',np.std((y-a@ls(a[m],y[m]))[m]),'matrix',ls(a[m],y[m]));
  if tag=='a':continue
  results=[]
  for n,f in fns().items():
   for kind in [0,1]:
    def design(z,k):return np.c_[a if kind else np.ones(len(z)),f(z,k)]
    def loss(k):
     a1=design(x,k);return np.mean((y-a1@ls(a1,y))**2)
    opt=minimize_scalar(loss,bounds=(.005,4),method='bounded'); k=opt.x; a1=design(x,k); h=np.clip(1-np.sum(a1*np.linalg.pinv(a1).T,axis=1),1e-6,None); cv=np.mean(((y-a1@ls(a1,y))/h[:,None])**2)**.5
    results.append((cv,n,kind,k,opt.fun**.5))
  print('best',sorted(results)[:8])
t=s['eval_times']; T=s['train']['traj'];print('times',t)
for tag in [0,1]:
 def base(r):return np.c_[(-np.expm1(-t[:,None]*r)), np.ones((len(t),1))] if tag else -np.expm1(-t[:,None]*r)
 r=least_squares(lambda r:(base(r)@ls(base(r),T[:52].T)-T[:52].T).ravel(),[2,.4],bounds=([.75,.01],[10,.75])).x
 print('time model',tag,r,'resid',np.mean(abs((base(r)@ls(base(r),T.T)-T.T))))
