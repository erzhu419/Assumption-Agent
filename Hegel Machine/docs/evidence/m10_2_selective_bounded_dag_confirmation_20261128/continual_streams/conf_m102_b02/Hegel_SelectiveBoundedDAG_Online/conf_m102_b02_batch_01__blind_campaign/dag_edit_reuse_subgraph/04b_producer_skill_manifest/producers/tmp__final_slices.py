import copy,importlib.util,pickle,sys
import numpy as np
sys.path.insert(0,'/tmp')
from node00_metric import _fit
from node00_baseline import run_node as baseline
p='/holdout_task/'
def load(i):
 s=importlib.util.spec_from_file_location(str(i),p+'_frozen_nodes/node_0'+str(i)+'.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.run_node
n1,n2=load(1),load(2)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; n=len(s['train']['x']); folds=[np.arange(52,72),*np.array_split(np.random.default_rng(407).permutation(n),4)]
r={}
for fold,te in enumerate(folds):
 tr=np.setdiff1d(np.arange(n),te); st=copy.deepcopy(s); st['train']={k:np.asarray(v)[tr] if np.ndim(v) and len(v)==n else v for k,v in s['train'].items()};st['test']['x']=s['train']['x'][te]
 r1=n1({'__input__':st})
 def ev(key,z):
  ans=n2({'node_00':z,'node_01':r1}); e=[np.mean(abs(ans['y_hat']-s['train']['y'][te])),np.mean(abs(ans['state_pred']-s['train']['traj'][te]))];r.setdefault(key,[]).append(e)
 ev((-1,0),baseline({'__input__':st}))
 for kind in range(3):
  arr,k=_fit(st,kind,7)
  for j,z in enumerate([arr[0],np.concatenate(arr[:3],axis=1)/np.sqrt(3),np.concatenate(arr,axis=1)/np.sqrt(len(arr))]):ev((kind,j),{'z':z[:k],'zt':z[k:]})
 print('fold',fold,{k:np.round(v[-1],5).tolist() for k,v in r.items()},flush=True)
print('SUMMARY',flush=True)
for k,v in sorted(r.items(),key=lambda kv: (max(kv[1][0])>.038,np.max(kv[1]))): print(k,'public',v[0],'worst',np.max(v[1:],axis=0),flush=True)
key=min(r,key=lambda k:(max(r[k][0])>.038,np.max(r[k])))
if key[0]>=0:
 source=open('/tmp/node00_metric.py').read().replace('KIND=0',f'KIND={key[0]}')
 if key[1]:
  count=3 if key[1]==1 else 7
  source=source.replace('z=options[0]',f'z=np.concatenate(options[:{count}],axis=1)/np.sqrt({count})')
 open(p+'node_00.py','w').write(source)
print('SELECTED',key,flush=True)
