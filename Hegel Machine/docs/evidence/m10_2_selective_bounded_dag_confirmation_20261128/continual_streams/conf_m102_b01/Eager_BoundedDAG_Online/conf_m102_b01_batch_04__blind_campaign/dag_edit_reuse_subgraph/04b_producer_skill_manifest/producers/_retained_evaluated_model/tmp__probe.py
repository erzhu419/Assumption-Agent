import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; w=run_node({'__input__':s})['w']; a,b,c=[s['panel_'+k] for k in 'abc']; tr=s['train']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; aff=lambda a:np.c_[np.ones(len(a)),a]; B=ls(aff(b['u']),b['v']); x=(w-B[0])@np.linalg.pinv(B[1:]); print('B',B,'B sv',np.linalg.svd(B[1:],compute_uv=False)); print('telemetry',s['telemetry_vec']); print('ranges',[(k,np.min(v,axis=0),np.max(v,axis=0)) for p in (b,c) for k,v in p.items()]); print('traj sv clean/noise',np.linalg.svd(tr['traj'],compute_uv=False)); print('cov y/traj',np.corrcoef(np.c_[tr['y'],tr['traj']].T)[0]); print('traj samples',tr['traj'][:8]);
def feats(x,d):return np.column_stack([np.prod(x[:,inds],axis=1) for q in range(d+1) for inds in itertools.combinations_with_replacement(range(4),q)])
def scan(u,v):
 for name,f in [('aff',aff),('poly2',lambda x:feats(x,2)),('odd cubic',lambda x:np.c_[aff(x),x**3]),('odd sin',lambda x:np.c_[aff(x),np.sin(1.25*x)]),('odd tanh',lambda x:np.c_[aff(x),np.tanh(1.25*x)]),('sin',lambda x:aff(np.sin(1.25*x))),('tanh',lambda x:aff(np.tanh(1.25*x))),('atan',lambda x:aff(np.arctan(1.25*x))),('asinh',lambda x:aff(np.arcsinh(1.25*x)))]:
  z=f(u); p=z@np.linalg.pinv(z); r=v-p@v; print(name,'in',np.round(np.mean(abs(r),axis=0),5),'LOO',np.round(np.mean(abs(r)/np.maximum(1-np.diag(p)[:,None],1e-9),axis=0),5))
for name,u,v in [('B',b['u'],b['v']),('C1',c['u'],c['v1']),('C2',c['u'],c['v2']),('C1 after B',aff(c['u'])@B,c['v1']),('C2 after C1',c['v1'],c['v2']),('C1 after C2',c['v2'],c['v1'])]:
 print(name); scan(u,v)
