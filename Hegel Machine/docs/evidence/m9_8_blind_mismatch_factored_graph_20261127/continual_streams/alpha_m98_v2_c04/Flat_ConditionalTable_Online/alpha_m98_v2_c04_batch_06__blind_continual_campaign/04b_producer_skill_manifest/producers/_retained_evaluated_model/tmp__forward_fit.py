import pickle,numpy as np
from itertools import combinations_with_replacement
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; c=d['calibration_load_cells']; R=c['bridge_raw']; L=c['true_channel_loads']
def terms(deg,kind):
 return [z for p in range(1,deg+1) for z in combinations_with_replacement(range(4),p) if kind=='all' or len(set(z))==1 or (kind=='pair' and p==2)]
def design(x,t): return np.column_stack([np.ones(len(x))]+[x[:,z].prod(1) for z in t])
def fit(a,r):
 w=np.ones(len(a))
 for j in range(35):
  b=np.linalg.lstsq(a*w[:,None],r*w[:,None],rcond=None)[0]; e=np.linalg.norm(a@b-r,axis=1); w=np.sqrt(np.minimum(1,max(0.00002,0.45*np.median(e))/np.maximum(e,1e-15)))
 good=e<max(5*np.median(e),.002)
 return np.linalg.lstsq(a[good],r[good],rcond=None)[0],good
for deg,kind in [(1,'all'),(2,'all'),(3,'pure'),(3,'pair'),(3,'all'),(4,'all')]:
 t=terms(deg,kind); a=design(L,t); b,g=fit(a,R); e=np.linalg.norm(a@b-R,axis=1); print(deg,kind,'inliers',g.sum(),'residual quantiles',np.round(np.quantile(e,[.5,.75,.85,.9,.95,1]),6)); print('largest modes',[(z,list(v.round(5))) for z,v in zip([()]+t,b) if np.max(abs(v))>.003] if deg==3 else '')
 if deg==3 and kind=='all':
  print('outlier rows',np.where(~g)[0]); print('residual singular values',np.linalg.svd((R-a@b)[~g],compute_uv=False)); print('outlier projections',np.round((R-a@b)[~g],4))
