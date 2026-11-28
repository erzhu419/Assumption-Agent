import pickle, numpy as np, itertools, importlib.util
from pathlib import Path
p=Path('/holdout_task'); s=importlib.util.spec_from_file_location('frozen',p/'_frozen_nodes/node_00.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
raw=pickle.load(open(p/'public_batch.pkl','rb')); print('batch', type(raw), len(raw)); study=raw[0].get('study',raw[0]); z=m.run_node({'__input__':study}); x=z['z_tr']; xt=z['z_te']; y=study['train']['y']; t=study['train']['traj']; n=len(y)
print('keys',raw[0].keys(),study.keys()); print('times',study['eval_times'],'summary',study['train']['summary'],'telemetry',study['telemetry_vec']); print('z train ranges',x.min(0),x.max(0),'test',xt.min(0),xt.max(0)); print('traj singular',np.linalg.svd(t,compute_uv=False)); print('sample z,y,traj'); print(np.column_stack((x,y,t))[:6]);
def poly(x,d):
    powers=[ix for k in range(1,d+1) for ix in itertools.combinations_with_replacement(range(x.shape[1]),k)]
    return np.column_stack([np.ones(len(x))]+[x[:,ix].prod(1) for ix in powers])
def loo(X,Y):
    B=np.linalg.lstsq(X,Y,rcond=None)[0]; e=Y-X@B; h=np.sum(X*np.linalg.pinv(X).T,axis=1); E=e/(1-h).reshape((-1,)+(1,)*(e.ndim-1)); return np.mean(abs(e),0), np.mean(abs(E),0), np.max(abs(E),0)
for d in (1,2,3):
    print('degree',d,'label',loo(poly(x,d),y),'traj',loo(poly(x,d),t))
for k in ('panel_b','panel_c'):
    a=study[k]; print(k,'u range',a['u'].min(0),a['u'].max(0));
    for v in a.keys()-{'u'}:
        print(v,'range',a[v].min(0),a[v].max(0),'linear/quad',loo(poly(a['u'],1),a[v])[:2],loo(poly(a['u'],2),a[v])[:2]); print('first rows', np.column_stack((a['u'],a[v]))[:3]);
