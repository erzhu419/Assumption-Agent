import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=150)
def design(u,d):
 return np.column_stack([np.ones(len(u))]+[u[:,c].prod(axis=1) for k in range(1,d+1) for c in combinations_with_replacement(range(u.shape[1]),k)])
def fit(x,y):
 c=np.linalg.lstsq(x,y,rcond=None)[0]; return c,np.linalg.norm(x@c-y,axis=1)
print('telemetry',s['telemetry_vec']);
for key in ['panel_a','panel_b','panel_c']:
 p=s[key]; u=p['u']; print(key,'u mean std range',u.mean(0),u.std(0),u.min(0),u.max(0)); print('u[:4]',u[:4]);
 for k in p.keys()-{'u'}:
  v=p[k]; print(k,'v[:4]',v[:4]);
  for d in (1,2):
   x=design(u,d); c,e=fit(x,v); lev=np.sum(np.linalg.qr(x)[0]**2,axis=1); print('degree',d,'rank',np.linalg.matrix_rank(x),'coef',c if d==1 else '', 'err',np.quantile(e,[0,.5,.85,.95,1]),'LOO',np.quantile(e/np.maximum(1-lev,1e-12),[.5,.85,1]));
   if d==1:
    keep=e<3*np.median(e); cr,er=fit(x[keep],v[keep]); print('refit',cr,'full err',np.sort(np.linalg.norm(x@cr-v,axis=1))); print('singular values',np.linalg.svd(cr[1:],compute_uv=False));
print('same panel u?',np.linalg.norm(s['panel_b']['u']-s['panel_c']['u'])); print('panel C difference',s['panel_c']['v2'][:6]-s['panel_c']['v1'][:6]); print('train x ranges',s['train']['x'].min(0),s['train']['x'].max(0));
