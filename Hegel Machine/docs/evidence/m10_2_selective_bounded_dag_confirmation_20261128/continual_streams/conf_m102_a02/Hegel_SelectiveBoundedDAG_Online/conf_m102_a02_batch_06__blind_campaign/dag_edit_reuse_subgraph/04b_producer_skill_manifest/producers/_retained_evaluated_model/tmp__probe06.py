import pickle, numpy as np, sys
from itertools import combinations_with_replacement as cwr
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as op
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; b=s['panel_b']; c=s['panel_c']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; z=op.run_node({'__input__':s}); print('frozen',z)
def design(x,d):
 return np.stack([np.prod(x[:,idx],axis=1) for deg in range(d+1) for idx in cwr(range(x.shape[1]),deg)],1)
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
for j in range(4):
 u,v=a['u'][:,j],a['v'][:,j]; i=np.argsort(u); print('A',j,np.round(np.c_[u,v,v-u*z['gain'][j]-z['offset'][j]][i],3).T)
for k in ('v1','v2'):
 for d in (1,2):
  f=design(c['u'],d); rr=f@ls(f,c[k])-c[k]; print('C',k,d,np.round(np.sqrt(np.mean(rr**2,axis=0)),5));
print('traj sigmas',np.linalg.svd(tr,compute_uv=False)); print('temporal SVD',np.linalg.svd(tr,full_matrices=False)[2][:3]);
q=(x-z['offset'])/z['gain']; st=op._b_features(q,z['p'])@z['w'];
for name,f in [('rawaff',design(x,1)),('rawq',design(x,2)),('baff',design(st,1)),('bq',design(st,2)),('all',np.c_[design(x,1),st])]:
 print(name,*(np.mean(abs(f[52:]@ls(f[:52],v[:52])-v[52:])) for v in (y,tr)))
