import numpy as np,pickle,glob,importlib.util
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(d,y):return np.linalg.lstsq(d,y,rcond=None)[0]
def pairs(x):return np.column_stack([x[:,i]*x[:,j] for i,j in cwr(range(x.shape[1]),2)])
def cal(panel,x):
 def design(z):
  q=np.roll(z,-1,axis=-1);return np.stack([np.ones_like(z),z,q,z*q,z**3],axis=-1)
 u,v=panel['u'],panel['v']; d=design(u); co=[]
 for j in range(4):
  a=d[:,j]; b=v[:,j]; ids=np.arange(len(a));
  for _ in range(15):
   w=ls(a[ids],b[ids]); e=abs(a@w-b);ids=np.argsort(e)[:40]
  ids=e<max(.001,8*np.median(e));co.append(ls(a[ids],b[ids]))
 co=np.array(co)
 from scipy.optimize import root
 def inv(row):
  fun=lambda z: (design(z)*co).sum(-1)-row
  r=root(fun,row,tol=1e-10)
  return r.x if max(abs(r.fun))<1e-8 else least_squares(fun,row,gtol=1e-12).x
 return np.array([inv(row) for row in x]).reshape(-1,4)
def dead(x):return np.sign(x)*np.maximum(abs(x)-.35,0)**1.5
b=s['panel_b']; c=s['panel_c'];u=c['u']; z=cal(s['panel_a'],np.r_[tr['x'],s['test']['x']]); w=aff(dead(z))@ls(aff(dead(b['u'])),b['v']); print('ranges',z.min(0),z.max(0),w.min(0),w.max(0))
from scipy.special import erf
u=c['u']; v=c['v1']; t=s['eval_times']; n=52

def score(q):
 y=tr['y']; p=tr['traj']; opp=np.c_[q[:,0]*q[:,2],q[:,1]*q[:,3]]
 ds=[aff(q),aff(np.c_[q,opp]),aff(np.c_[q,pairs(q)]),aff(np.c_[q,w]),aff(np.c_[q,w,opp])]
 return np.round([[np.mean(abs(d[n:72]@ls(d[:n],a[:n])-a[n:72])) for d in ds] for a in (y,p)],6)
for name,f in {'tanh':lambda x,k:np.tanh(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'atan':lambda x,k:np.arctan(k*x),'erf':lambda x,k:erf(k*x),'sqrt':lambda x,k:x/np.sqrt(1+k*x*x)}.items():
 for k in [1.25,1.8,2.15]:
  d=aff(np.c_[u,f(u,k)]); co=ls(d,v); print(name,k,'C1 free rmse',np.std(d@co-v),'linear coeff', np.round(co[1:5],4) if k==2.15 else ''); q=aff(np.c_[w,f(w,k)])@co; print('score free',score(q).tolist());
  if k==2.15:
   for kind in range(4):
    if kind==0:
     rhs=v-.22*u; des=lambda x:aff(f(x,k)); off=lambda x:.22*x
    elif kind==1:
     rhs=v; des=lambda x:aff(f(x,k)+.22*x); off=lambda x:0
    elif kind==2:
     rhs=v-.22*f(u,k); des=aff; off=lambda x:.22*f(x,k)
    else:
     rhs=v; des=lambda x:aff(f(x,k)+.22*np.roll(x,-1,axis=1)); off=lambda x:0
    b1=ls(des(u),rhs); q=des(w)@b1+off(w); print('structured',kind,np.std(des(u)@b1-rhs),score(q).tolist())
print('C2')
for name,f in {'square':lambda x,k:x+k*x*x,'cube':lambda x,k:x+k*x*x+k*x**3,'abs':lambda x,k:x+k*abs(x)*x,'tanhmix':lambda x,k:x+k*np.tanh(1.25*x)**2}.items():
 opt=minimize_scalar(lambda k:np.std(aff(f(u,k))@ls(aff(f(u,k)),c['v2'])-c['v2']),bounds=(-1,1),method='bounded');print(name,opt.x,opt.fun)
print('temporal')
p=tr['traj'][:52];
for mode in ['two_exp','exp_lin','exp_slow','exp_quad','exp_log']:
 def d(r):
  a=-np.expm1(-r[0]*t); v={'two_exp':-np.expm1(-r[-1]*t),'exp_lin':t,'exp_slow':-np.expm1(-.22*t),'exp_quad':t*t,'exp_log':np.log1p(t)}[mode];return np.c_[a,v]
 opt=least_squares(lambda r:(d(r)@ls(d(r),p.T)-p.T).ravel(),[1.25,.22] if mode=='two_exp' else [1.25],bounds=(.001,10)); print(mode,opt.x,np.std(opt.fun))
