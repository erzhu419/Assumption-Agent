import importlib.util
import pickle
import numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('candidate',path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m=load('/holdout_task/node_01.py')
up=load('/holdout_task/_frozen_nodes/node_00.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
f=up.run_node({'__input__':s})
t=s['train']; n=len(t['x']); ids=np.arange(n)
q=m._calibrate(s['panel_a'],f,t['x'])
rng=np.random.default_rng(135)
splits={f'block_{j}':ids[j:j+18] for j in (0,18,36,54)}
splits.update(public=ids[52:], interleaved=ids[::4], outer_radius=np.argsort(np.max(abs(q),axis=1))[-15:], extreme_square_difference=np.argsort(abs(q[:,0]**2-q[:,2]**2))[-15:])
splits.update({f'random_{j}':rng.choice(ids,18,False) for j in range(16)})
rows=[]
for name,j in splits.items():
    i=np.setdiff1d(ids,j)
    sf={**s,'train':{k:t[k][i] for k in ('x','y','traj')},'test':{'x':t['x'][j]}}
    result=m.run_node({'__input__':sf,'node_00':f})
    y,st,d=(result[k] for k in ('y_hat','state_pred','diagnostics_vec'))
    assert y.shape==(len(j),) and st.shape==(len(j),t['traj'].shape[1]) and d.shape==(4,)
    assert np.all(np.isfinite(st)) and np.all(np.isfinite(y))
    assert np.allclose(d,[y.mean(),y.std(),st[:,-1].mean(),st[:,-1].std()])
    rows.append((name,float(np.mean(abs(y-t['y'][j]))),float(np.mean(abs(st-t['traj'][j]))),float(np.max(np.mean(abs(st-t['traj'][j]),axis=0)))))
for row in rows: print('%s %.6f %.6f %.6f'%row)
print('worst scalar',max(rows,key=lambda row:row[1])); print('worst state and horizon',max(r[2] for r in rows),max(r[3] for r in rows))
assert max(max(r[1:]) for r in rows)<.038
