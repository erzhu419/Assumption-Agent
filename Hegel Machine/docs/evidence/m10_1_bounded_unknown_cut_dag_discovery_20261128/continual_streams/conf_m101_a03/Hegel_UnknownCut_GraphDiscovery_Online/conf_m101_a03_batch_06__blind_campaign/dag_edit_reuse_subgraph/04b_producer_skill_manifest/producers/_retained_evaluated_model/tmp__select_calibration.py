import pickle, importlib.util, numpy as np
from pathlib import Path
ns={}; core=Path('/tmp/calibration_core.py').read_text(); exec(core, ns)
def frozen(name):
 p=f'/holdout_task/_frozen_nodes/{name}.py'; spec=importlib.util.spec_from_file_location(name,p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod);return mod.run_node
operator, readout=map(frozen,('node_01','node_02'))
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); rng=np.random.default_rng(429)
# Refit on every split, including the panel coefficients and both latent indices.
orders=[np.arange(n)]+[rng.permutation(n) for _ in range(4)]; results=[]
for mode in range(5):
 for chart in range(3):
  errors=[]
  for j in orders:
   a,b=j[:-20],j[-20:]; t={**s,'train':{k:tr[k][a] for k in ('x','y','traj')},'test':{'x':tr['x'][b]}}
   cal=ns['_coordinates']({'__input__':t},mode,chart); y=readout({'node_01':operator({'__input__':t,'node_00':cal})}); errors.append([np.mean(abs(y['y_hat']-tr['y'][b])),np.mean(abs(y['state_pred']-tr['traj'][b]))])
  es=np.array(errors); print(mode,chart,np.round(es,5).tolist(),flush=True)
  results.append((np.max(es[0])>.038,np.max(es),np.mean(es),mode,chart))
_,worst,_,mode,chart=min(results)
print('selected',mode,chart,'worst',worst,flush=True)
Path('/holdout_task/node_00.py').write_text(core+f'\n\ndef run_node(inputs: dict) -> dict:\n    return _coordinates(inputs, {mode}, {chart})\n')
