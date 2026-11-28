import pickle,numpy as np
from itertools import combinations_with_replacement as cr
from scipy.linalg import svd
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x): return np.c_[np.ones(len(x)),x]
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def terms(x,d): return np.column_stack([np.prod(x[:,j],axis=1) for j in cr(range(x.shape[1]),d)])
a,b,c=[s['panel_'+k] for k in 'abc']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times'];
for j in range(4):
 f=aff(a['u'][:,j,None]); v=a['v'][:,j]; m=np.argsort(abs(v-a['u'][:,j]))[:39];
 for i in range(10):
  h=fit(f[m],v[m]); m=np.argsort(abs(v-f@h))[:39]
 print('cal',j,h, np.sort(abs(v-f@h))[::3]);
for v in [c['v1'],c['v2']]:
 u=c['u'];
 for g in [aff(u),np.c_[aff(u),u*u],np.c_[aff(u),terms(u,2)],np.c_[aff(u),terms(u,2),u**3]]:
  h=fit(g,v); r=v-g@h; loo=r/(1-(g*np.linalg.pinv(g).T).sum(1))[:,None]; print('panel',g.shape[1],np.sqrt((r*r).mean()),np.sqrt((loo*loo).mean()));
  if g.shape[1]==15: print(h.round(4))
_,sv,w=svd(z,full_matrices=False); print('sv',sv,'time basis',w[:2]);
for rg in [x,terms(x,2),terms(x,3)]: print('moments',np.max(abs(np.corrcoef(np.c_[rg,y,z@w[:2].T].T)[-3:,:-3]),1))
