import pickle,runpy,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s}); q=z['cubic_train']; b=s['panel_b']; u,v=b['u'],b['v']; y,t=s['train']['y'],s['train']['traj']; n,d=q.shape

def aff(x): return np.c_[np.ones(len(x)),x]
def mat(x,deg):
 if deg==1:return aff(x)
 return np.c_[aff(x),x[:,np.triu_indices(d)[0]]*x[:,np.triu_indices(d)[1]]]
def fit(f):
 a=aff(np.cos(f*u)); co=np.linalg.lstsq(a,v,rcond=None)[0]; return np.mean((a@co-v)**2),co
f=minimize_scalar(lambda f:fit(f)[0],bounds=(2.4,2.8),method='bounded').x; co=fit(f)[1]; print('frequency',f,'coef',co); print('traj singular',np.linalg.svd(t,compute_uv=False)); print('q range',q.min(0),q.max(0))
for name,p in [('cubic',q),('latent',np.cbrt(q)),('cubic cosine',np.cos(f*q)),('latent cosine',np.cos(f*np.cbrt(q))),('cubic panel',aff(np.cos(f*q))@co),('latent panel',aff(np.cos(f*np.cbrt(q)))@co)]:
 for degree in [1,2]:
  a=mat(p,degree); stats=[]
  for out in [y,t]:
   err=[]; oof=np.zeros_like(out)
   for held in np.array_split(np.random.default_rng(42).permutation(n),6):
    tr=np.setdiff1d(np.arange(n),held); pred=a[held]@np.linalg.lstsq(a[tr],out[tr],rcond=None)[0]; oof[held]=pred; err.append(np.mean(np.abs(pred-out[held])))
   stats.extend((float(np.mean(np.abs(oof-out))),float(max(err))))
  print(name,degree,np.round(stats,6))
