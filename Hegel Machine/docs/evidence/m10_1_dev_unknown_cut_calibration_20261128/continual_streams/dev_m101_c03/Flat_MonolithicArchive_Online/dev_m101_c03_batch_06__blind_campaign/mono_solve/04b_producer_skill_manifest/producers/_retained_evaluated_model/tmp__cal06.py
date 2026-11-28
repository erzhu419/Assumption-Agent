import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def calterms(z,shift=-1,kind=0):
 w=np.roll(z,shift,axis=-1)
 cols=[np.ones_like(z),z,w,z*w,z**3]
 if kind==1:cols += [w*w,z*z,z*z*w,z*w*w,w**3]
 if kind==2:cols += [np.roll(z,-2,axis=-1),z*np.roll(z,-2,axis=-1),w*w,z*z,z*z*w,z*w*w,w**3]
 return np.stack(cols,axis=-1)
def calibrate(kind=0,shift=-1):
 u,v=s['panel_a'].values();f=calterms(u,shift,kind);rng=np.random.default_rng(8311);best=(np.inf,None,None)
 def fit(ix):return np.stack([ls(f[ix,j],v[ix,j]) for j in range(4)])
 for _ in range(150):
  ix=rng.choice(len(u),max(f.shape[-1]+2,9),False)
  for k in range(5):
   a=fit(ix);err=np.linalg.norm(np.sum(f*a,axis=-1)-v,axis=1);ix=np.argsort(err)[:40]
  e=np.mean(err[ix]**2)**.5
  if e<best[0]:best=e,a,err
 return best
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True)
 for shift in (-1,1):
  for kind in (0,1,2):
   e,a,r=calibrate(kind,shift);print('A',shift,kind,e,'ordered',np.sort(r)[35:],'coef',a)
 for name,a in [('train',s['train']['x']),('test',s['test']['x'])]: print(name,'x quantiles',np.quantile(a,[0,.05,.5,.95,1],axis=0))
