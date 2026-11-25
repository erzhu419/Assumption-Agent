import pickle,sys,time
import numpy as np
sys.path.insert(0,'/holdout_task')
from certified_box_l0 import solve_certified_box_l0
from box_l0_baseline import solve_certified_box_l0 as baseline
cases = pickle.load(open('/holdout_task/public_box_l0_cases.pkl','rb'))['cases']
rng=np.random.default_rng(94)
for p,ridge,M in [(7,0.,.24),(8,.07,.75),(8,0.,.3),(14,0.,.18),(15,.025,.25),(15,0.,.35),(16,.04,.5),(15,0.,.05),(15,0.,.10)]:
 n=34; latent=rng.normal(size=(n,4)); X=latent@rng.normal(size=(4,p))+.16*rng.normal(size=(n,p)); X[:,1]=X[:,0]; X[:,3]=-2*X[:,2]; y=X[:,::2]@rng.normal(size=(p+1)//2)+rng.normal(size=n)*.18
 cases.append(dict(X=X,y=y,ridge=ridge,l0_penalty=.18,box=M,gap_tol=1e-6,max_leaves=512))
for i,d in enumerate(cases):
 t=time.perf_counter(); r=solve_certified_box_l0(d); dt=time.perf_counter()-t
 t=time.perf_counter(); ref=baseline(d); old=time.perf_counter()-t
 w=np.asarray(r['weights']); e=d['X']@w-d['y']; j=.5*(e@e+d['ridge']*(w@w))+d['l0_penalty']*np.count_nonzero(abs(w)>1e-10); lb=r['certified_lower_bound']
 assert np.all(abs(w)<=d['box']) and abs(j-r['objective'])<=1e-7 and 0<=j-lb<=d['gap_tol']
 assert lb<=ref['objective']+1e-8 and abs(j-ref['objective'])<=d['gap_tol']
 assert r['certificate_leaves']==[dict(fixed_zero=[],fixed_nonzero=[],leaf_lower_bound=lb)]
 print(i, 'seconds new/old',round(dt,3),round(old,3),'gap',j-lb,flush=True)
