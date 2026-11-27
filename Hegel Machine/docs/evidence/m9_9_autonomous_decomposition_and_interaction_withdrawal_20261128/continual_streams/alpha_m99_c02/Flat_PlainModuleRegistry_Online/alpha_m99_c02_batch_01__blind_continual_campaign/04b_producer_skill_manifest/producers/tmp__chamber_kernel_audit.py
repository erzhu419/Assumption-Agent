import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
s=runpy.run_path('/holdout_task/batch_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d,g=i['dataset'],i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']
b=s['_bridge'](d['calibration_load_cells']); x,_=s['_features'](tr,te); u,_=s['_features'](g,g)
q,w=s['_material'](tr,b,x,dt,t); _,h=s['_loads'](g,b); lab=g['lab_id']; drift=w[2]+u@w[3:]
y=g['strain_curves_obs']-g['strain_0'][:,None]; load=(h[:,None]*w[:2])@(-np.expm1(-t[:,None]/q)).T
sat=lambda a:-a*np.expm1(-t/a)
def prediction(f,v): return load+drift[:,None]*(f*sat(q[0])+(1-f)*sat(v))
def solve(mode,m):
 if mode=='free':
  return least_squares(lambda a:(prediction(*a)[m]-y[m]).ravel(),[.55,5],bounds=([0,q[0]],[1,25])).x
 if mode=='fraction':
  return [least_squares(lambda a:(prediction(a[0],5)[m]-y[m]).ravel(),[.55],bounds=([0],[1])).x[0],5]
 if mode=='slow':
  return [.55,least_squares(lambda a:(prediction(.55,a[0])[m]-y[m]).ravel(),[5],bounds=([q[0]],[25])).x[0]]
 if mode=='intrinsic':
  return [least_squares(lambda a:(prediction(a[0],q[1])[m]-y[m]).ravel(),[.55],bounds=([0],[1])).x[0],q[1]]
 return [.55,5]
rm=lambda a:np.sqrt(np.mean(a*a))
for mode in ['old','fraction','slow','intrinsic','free']:
 e=np.empty_like(y); params=[]
 for l in np.unique(lab):
  m=lab!=l; a=solve(mode,m); params.append(a); e[~m]=prediction(*a)[~m]-y[~m]
 complete=np.isfinite(g['coupon_features_raw']).all(1)
 print(mode,'full',np.round(solve(mode,np.ones(len(lab),bool)),7),'fold params',np.round(params,5).tolist(),'worst lab',round(max(rm(e[lab==l]) for l in np.unique(lab)),7),'late',round(rm(e[:,-3:]),7),'complete/missing',np.round([rm(e[complete]),rm(e[~complete])],7))
