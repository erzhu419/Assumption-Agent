import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; n=len(x)
np.set_printoptions(precision=5,suppress=True)
folds=np.random.default_rng(20).permutation(n)%5

def lin(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def robust(a,b):
 m=lin(a,b)
 for scale in [.3,.1,.03,.01]:
  for _ in range(12):
   w=1/np.hypot(1,np.linalg.norm(a@m-b,axis=1)/scale); m=lin(a*w[:,None],b*w[:,None])
 return m

def sat(z,k,kind):
 if kind=='rat': return z/(1+k*z*z)
 if kind=='root': return z/np.sqrt(1+k*z*z)
 if kind=='tanh': return np.tanh(z*np.sqrt(k))/np.sqrt(k)
 if kind=='atan': return np.arctan(z*np.sqrt(k))/np.sqrt(k)
 if kind=='exp': return z*np.exp(-k*z*z)

def fit(x,y,kind,layout):
 af=lambda z: np.column_stack((np.ones(len(z)),z))
 a=af(x); base=robust(a,y); e=np.linalg.norm(a@base-y,axis=1); good=e<max(2,4*np.median(e))
 if layout in ('pre','preplus','prequad'):
  def des(z,k):
   q=sat(z,k,kind)
   return af(q) if layout=='pre' else np.column_stack((af(q),z) if layout=='preplus' else (af(q),z,z*z))
  p=np.full(4,.7)
  for _ in range(3):
   def err(k):
    a=des(x[good],k); return (a@lin(a,y[good])-y[good]).ravel()
   p=least_squares(err,p,bounds=(.003,15),max_nfev=100).x
   a=des(x,p); m=lin(a[good],y[good]); e=np.linalg.norm(a@m-y,axis=1); good=e<max(.025,5*np.median(e))
  return lambda z:des(z,p)@m, p, m
 # Each channel is a nonlinear function of a normalized row of the mixing map.
 p=np.r_[base[0],(base[1:]/np.diag(base[1:]))[~np.eye(4,dtype=bool)],np.diag(base[1:]),np.full(4,.7),np.full(4,0.) if layout=='postplus' else []]
 def pred(z,p):
  mix=np.eye(4); mix[~np.eye(4,dtype=bool)]=p[4:16]; h=z@mix
  return p[:4]+p[16:20]*sat(h,p[20:24],kind)+(p[24:28]*h if layout=='postplus' else 0)
 for _ in range(3):
  b=np.full(len(p),np.inf); low=-b.copy(); low[20:24]=.003; b[20:24]=15
  p=least_squares(lambda p:(pred(x[good],p)-y[good]).ravel(),p,bounds=(low,b),max_nfev=120).x
  e=np.linalg.norm(pred(x,p)-y,axis=1); good=e<max(.025,5*np.median(e))
 return lambda z:pred(z,p),p,None

a=np.column_stack((np.ones(n),x)); base=robust(a,y); e=np.linalg.norm(a@base-y,axis=1); good=e<2
print('EMI rows:',np.flatnonzero(~good), 'affine worst fold med',max(np.median(np.linalg.norm(a[(folds==j)&good]@robust(a[folds!=j],y[folds!=j])-y[(folds==j)&good],axis=1)) for j in range(5)))
for layout in ('pre','preplus','prequad','post','postplus'):
 for kind in ('rat','root','tanh','atan','exp'):
  f,p,m=fit(x,y,kind,layout); r=np.linalg.norm(f(x)-y,axis=1)
  vals=[]
  for j in range(5):
   q=folds==j; ff,_,_=fit(x[~q],y[~q],kind,layout); v=np.linalg.norm(ff(x[q&good])-y[q&good],axis=1); vals.append([np.median(v),np.max(v)])
  print(layout,kind,'fit med',round(np.median(r[good]),7),'CV worst med/max',np.max(vals,axis=0),'params',p if m is None else p.tolist(),flush=True)
  if np.median(r[good])<.03:
   print('coef',m,'clean max',max(r[good]),'residual rows',np.c_[x,y-f(x)][np.argsort(r)[-len(x[~good])-5:-len(x[~good])]],flush=True)
