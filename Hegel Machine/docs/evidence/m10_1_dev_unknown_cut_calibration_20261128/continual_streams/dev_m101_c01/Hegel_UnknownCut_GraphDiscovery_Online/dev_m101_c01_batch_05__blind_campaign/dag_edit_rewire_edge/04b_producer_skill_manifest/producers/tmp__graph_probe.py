import numpy as np, pickle, runpy
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; raw=np.vstack([s['train']['x'],s['test']['x']]); x=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['x']; u,v=s['panel_b']['u'],s['panel_b']['v']; b=np.linalg.lstsq(np.c_[np.ones(len(u)),u],v,rcond=None)[0]; print('B singular values',np.linalg.svd(b[1:],compute_uv=False)); print('train x ranges',np.round(raw.min(0),2),np.round(raw.max(0),2),'cal',np.round(x.min(0),2),np.round(x.max(0),2)); u=s['panel_c']['u'];
for key in ('v1','v2'):
 V=s['panel_c'][key]; print(key)
 for f in (np.tanh,np.sin,np.cos):
  for design in ('identity','pair','rotB','invB'):
   w={'identity':u,'pair':np.c_[u[:,0]+u[:,1],u[:,0]-u[:,1],u[:,2]+u[:,3],u[:,2]-u[:,3]],'rotB':u@b[1:],'invB':u@np.linalg.inv(b[1:])}[design]
   def err(a):
    D=np.c_[np.ones(len(u)),u,f(w*a[:4]+a[4:])]; return (D@np.linalg.lstsq(D,V,rcond=None)[0]-V).ravel()
   out=least_squares(err,np.r_[np.full(4,1.5),np.zeros(4)],max_nfev=100); print(f.__name__,design,round(np.std(out.fun),6),np.round(out.x,3))
Y=np.c_[s['train']['y'],s['train']['traj']]; rng=np.random.default_rng(18); folds=[np.arange(52,72)]+np.array_split(rng.permutation(72),4); n=len(Y)
def eval_model(C):
 vals=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); F=C[:n]; pr=F[te]@np.linalg.lstsq(F[tr],Y[tr],rcond=1e-9)[0]; e=np.abs(pr-Y[te]); vals.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()])
 return np.round(vals,4)
for tag,w in [('raw',raw),('cal',x),('raw-invA',raw@np.linalg.lstsq(np.c_[np.ones(len(x)),x],raw,rcond=None)[0][1:]),('cal-B',x@b[1:]+b[0]),('cal-invB',(x-b[0])@np.linalg.inv(b[1:]))]:
 P=lambda d:np.array([np.prod(w[:,ii],axis=1) for ii in cwr(range(4),d)]).T
 for deg in (1,2,3): print(tag,deg,eval_model(np.c_[np.ones(len(w)),*(P(j) for j in range(1,deg+1))]).tolist())
 for f in (np.tanh,np.sin):
  z=f(1.7*w); print(tag,f.__name__,eval_model(np.c_[np.ones(len(w)),P(1),P(2),z,np.array([np.prod(z[:,ii],axis=1) for ii in cwr(range(4),2)]).T]).tolist())
