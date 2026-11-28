import pickle, runpy, numpy as np
from pathlib import Path
p=Path('/holdout_task'); ns=runpy.run_path('/tmp/final_candidate.py'); sink=runpy.run_path(str(p/'_frozen_nodes/node_01.py'))['run_node']; s=pickle.load((p/'public_batch.pkl').open('rb'))[0]['study']; tr=s['train']; ix=np.arange(len(tr['x'])); cut=max(40,len(ix)-20); prefix=ix[:cut]; suffix=ix[cut:]; folds=np.array_split(np.random.default_rng(351).permutation(prefix),4)

def evaluate(a,b):
    st=dict(s,train={k:tr[k][a] for k in ('x','y','traj')},test={'x':tr['x'][b]}); inp={'__input__':st}; data=ns['_previous_node'](inp); h=ns['_structured_features'](data,np.vstack((tr['x'][a],tr['x'][b]))); out={}
    for m in (0,1,3,8):
        r=sink({'node_00':dict(data,z=np.column_stack((data['z'],np.tile(h,(1,m)))))}); out[m]=np.array([np.mean(abs(r['y_hat']-tr['y'][b])),np.mean(abs(r['state_pred']-tr['traj'][b]))])
    return out

inner=[evaluate(np.setdiff1d(prefix,b),b) for b in folds]
worst={m:np.max([e[m] for e in inner],axis=0) for m in inner[0]}; print('inner worst MAEs',worst,flush=True)
m=min(worst,key=lambda m:max(worst[m])); outer=evaluate(prefix,suffix)
print('official suffix MAEs, baseline and inner-selected challenger',outer[0],m,outer[m],flush=True)
if m and (max(outer[m]) <= .038 or (np.all(outer[m] <= outer[0]) and max(outer[m]) < max(outer[0]))):
    with (p/'node_00.py').open('w') as f:
        f.write(Path('/tmp/final_candidate.py').read_text()+f'\n\ndef run_node(inputs: dict) -> dict:\n    return _transfer(inputs, {m})\n')
