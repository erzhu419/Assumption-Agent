import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; rng=np.random.default_rng(311)
folds=[(np.arange(52),np.arange(52,72))]
for _ in range(5):
 p=rng.permutation(72);folds.append((p[:52],p[52:]))

def poly(x,d):
 return np.c_[np.ones(len(x)),*[np.prod(x[:,c],axis=1) for k in range(1,d+1) for c in combinations_with_replacement(range(4),k)]]
def fit_predict(x,xt,y,degree,acts):
 p,pt=poly(x,degree),poly(xt,degree); q=np.linalg.qr(p)[0]; resid=y-q@(q.T@y)
 def hidden(x,z):
  a=np.c_[x,np.ones(len(x))]@z.reshape(-1,5).T
  return np.stack([f(a[:,j]) for j,f in enumerate(acts)],axis=1)
 def fun(z,ret=False):
  h=hidden(x,z)
  if ret:
   a=np.c_[p,h]; c=np.linalg.lstsq(a,y,rcond=1e-9)[0]; return np.c_[pt,hidden(xt,z)]@c
  h-=q@(q.T@h); c=np.linalg.lstsq(h,resid,rcond=1e-9)[0]
  return np.r_[resid-h@c, .001*z]
 starts=[np.r_[np.eye(4)[k],b] for k in range(4) for b in (0.,.6)]
 rr=np.random.default_rng(773)
 starts+=list(rr.normal(0,.75,(20,5)))
 ini=[np.concatenate([starts[(k+7*j)%len(starts)] for j in range(len(acts))]) for k in range(len(starts))]
 results=[least_squares(fun,a,bounds=(-4,4),max_nfev=105) for a in ini]
 results=sorted(results,key=lambda r: np.linalg.norm(r.fun))
 preds=np.array([fun(r.x,True) for r in results[:5]])
 return preds[0],np.median(preds,axis=0),np.sqrt(np.mean(results[0].fun[:len(x)]**2)),results[0].x
acts={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rad':lambda z:1/(1+z*z),'rat':lambda z:z/(1+z*z),'cos':np.cos}
for degree, names in [(1,('tanh',)),(1,('tanh','tanh')),(1,('sin','tanh')),(2,('tanh',)),(2,('sin',)),(2,('atan',)),(2,('rad',)),(2,('rat',)),(2,('cos',)),(2,('tanh','sin')),(2,('sin','sin')),(2,('tanh','tanh')),(2,('rad','sin'))]:
 errors=[]; fits=[]
 for i,j in folds:
  p,m,e,z=fit_predict(x[i],x[j],y[i],degree,[acts[k] for k in names]); errors.append([abs(p-y[j]).mean(),abs(m-y[j]).mean(),e]);fits.append(z)
 e=np.array(errors); print(degree,names,'split',e[0].round(5),'mean',e.mean(0).round(5),'max',e.max(0).round(5),'coeffs',fits[0].round(2),flush=True)
