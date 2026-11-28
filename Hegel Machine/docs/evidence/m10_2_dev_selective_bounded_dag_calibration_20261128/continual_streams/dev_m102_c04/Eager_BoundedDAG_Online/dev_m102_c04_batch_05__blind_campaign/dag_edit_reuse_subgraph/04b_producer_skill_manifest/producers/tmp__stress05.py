import importlib.util,pickle,numpy as np

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
root='/holdout_task/'
m=load(root+'node_00.py','spatial'); t=load(root+'_frozen_nodes/node_01.py','temporal'); f=load(root+'_frozen_nodes/node_02.py','sink')
for entry in pickle.load(open(root+'public_batch.pkl','rb')):
 s=entry['study'];tr=s['train']; idx=np.arange(len(tr['x'])); x=tr['x']; y=tr['y']; z=tr['traj']; q=m._inverse(s['panel_a'],x); full=m._inverse(s['panel_a'],np.vstack((x,s['test']['x']))); print('calibrated range',full.min(),full.max())
 def check(v):
    keep=np.setdiff1d(idx,v); tt=z[keep]; su=np.array([np.mean(y[keep]),np.std(y[keep]),np.mean(tt[:,-1]),np.std(tt[:,-1])]); ss={**s,'train':{'x':x[keep],'y':y[keep],'traj':tt,'summary':su},'test':{'x':x[v]}}; payload={'__input__':ss}; a=m.run_node(payload);out=f.run_node({'node_00':a,'node_01':t.run_node(payload)});ey=np.abs(out['y_hat']-y[v]);et=np.abs(out['state_pred']-z[v]); return np.r_[ey.mean(),et.mean(),et.mean(0).max()]
 for label,folds in [('row blocks',np.array_split(idx,4)),('spatial tails',[np.argsort(q[:,j])[sl] for j in range(q.shape[1]) for sl in (slice(0,16),slice(-16,None))]),('random splits',[np.random.default_rng(seed).permutation(idx)[:20] for seed in range(8)])]:
    metrics=np.array([check(v) for v in folds]);print(label,'worst [y, traj, worst horizon]',metrics.max(0))
