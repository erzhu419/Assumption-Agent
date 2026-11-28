import sys,pickle,numpy as np,itertools
from importlib.util import spec_from_file_location,module_from_spec
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_02
sp=spec_from_file_location('candidate','/holdout_task/node_01.py'); m=module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=len(s['train']['y']); idx=np.arange(n)
folds=[idx[-20:]]+list(np.array_split(idx,4))+[np.random.default_rng(i).permutation(idx)[:20] for i in range(3)]
data=[]
for j in folds:
 i=np.setdiff1d(idx,j); st=dict(s,train={k:s['train'][k][i] for k in ('x','y','traj')},test={'x':s['train']['x'][j]}); p=node_00.run_node({'__input__':st}); state=node_02.run_node({'node_00':p})['state_pred']; data.append((p,state,s['train']['y'][j]))
res=[]
for kind,scale,mix,t in itertools.product(['st','sy','all','K'],[.025,.06,.12,.24],[0,1,10,100],[0,.015,.12]):
 hh=[(*m._basis(p,q,kind,scale,mix,t),p['y_centered'],p['y_mean'],y) for p,q,y in data]
 for l,c in itertools.product([.0003,.003,.03,.15],[0,2,3]):
  vals=np.array([np.mean(abs(m._predict(h,hq,y,l,c)+mu-truth)) for h,hq,y,mu,truth in hh]); res.append((max(vals),np.mean(vals),vals[0],(kind,scale,mix,t,l,c),vals.round(5)))
print('MINIMAX');
for v in sorted(res,key=lambda r:r[0])[:12]: print(v)
print('OFFICIAL');
for v in sorted(res,key=lambda r:r[2])[:8]: print(v)
