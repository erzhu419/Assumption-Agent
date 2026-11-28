import pickle,sys,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as F
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=F.run_node({'__input__':s}); tr=s['train']; x=(tr['x']-f['offset'])/f['gain']; y=tr['y']; z=tr['traj']; b=F._b_features(x,f['p'])@f['w']; print('gain offset p w',*(f[k] for k in f)); print('x quantiles',np.quantile(x,[0,.05,.5,.95,1],axis=0)); print('panel a residual quantiles',np.quantile(abs(s['panel_a']['v']-s['panel_a']['u']*f['gain']-f['offset']),[.5,.9,.95,.98,1],axis=0)); print('traj spectrum',np.linalg.svd(z,compute_uv=False))
def poly(a,d):
 return np.column_stack([np.ones(len(a))]+[np.prod(a[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(a.shape[1]),k)])
def report(n,a,o):
 p=a@np.linalg.lstsq(a[:52],o[:52],rcond=None)[0]; print(n,a.shape[1],np.mean(abs(p[:52]-o[:52])),np.mean(abs(p[52:]-o[52:])),np.max(abs(p[52:]-o[52:])))
for n,a in [('x',x),('b',b),('xb',np.c_[x,b])]:
 for d in range(1,4 if n!='xb' else 3):
  v=poly(a,d); report('y '+n+str(d),v,y); report('z '+n+str(d),v,z)
for r in [1,2]:
 u=s['panel_c']['u']; v=s['panel_c'][f'v{r}']; print('panel c',r,'sv',np.linalg.svd(v,compute_uv=False));
 for d in (1,2):
  a=poly(u,d); c=np.linalg.lstsq(a[:19],v[:19],rcond=None)[0]; print(d,np.mean(abs(a@c-v),axis=0),np.mean(abs(a[19:]@c-v[19:])))
print('panel b err',np.mean(abs(F._b_features(s['panel_b']['u'],f['p'])@f['w']-s['panel_b']['v'])))
