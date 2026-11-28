from readout_probe import *
import importlib.util

def load(name, path):
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.run_node

base=load('baseline','/holdout_task/node_01.py')
new=load('challenger','/tmp/readout_challenger.py')
def score(ds):
 e=[]
 for p,pr,y,tr in ds:
  a={'node_00':p,'node_02':{'state_pred':pr}}
  e.append([np.abs(fn(a)['y_hat']-y).mean() for fn in (base,new)]+[np.abs(pr-tr).mean()])
 e=np.asarray(e);print(np.round(e,6)); print('mean',e.mean(0),'worst',e.max(0)); return e
print('DEVELOPMENT: baseline/challenger/trajectory'); score(batches())
x=s['train']['x']; groups=np.concatenate([np.argsort(x[:,j])[:14][None,:] for j in range(x.shape[1])]); random=np.array_split(np.random.default_rng(372).permutation(n),6)
def holdouts(indices):
 for I in indices:
  J=np.setdiff1d(np.arange(n),I); z=copy.deepcopy(s);z['train']={k:v[J] if len(v)==n else v for k,v in s['train'].items()};z['test']['x']=x[I]
  p=node_00.run_node({'__input__':z}); pr=node_02.run_node({'node_00':p})['state_pred'];yield p,pr,s['train']['y'][I],s['train']['traj'][I]
print('ALTERNATE RANDOM');score(holdouts(random))
print('LOW COORDINATE TAILS');score(holdouts(groups))
