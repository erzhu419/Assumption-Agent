import copy, importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('_candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=len(x)
def evaluate(idx, panel=None, drop=None):
    keep=np.setdiff1d(np.arange(n),idx); q=copy.deepcopy(s)
    q['test']['x']=x[idx]; q['train'].update({k:tr[k][keep] for k in ('x','y','traj')})
    q['train']['summary']=np.array([q['train']['y'].mean(),q['train']['y'].std(),q['train']['traj'][:,-1].mean(),q['train']['traj'][:,-1].std()])
    if panel: q[panel]={k:np.delete(v,drop,axis=0) for k,v in q[panel].items()}
    z=m.solve_batch(q); dy=abs(z['y_hat']-tr['y'][idx]); dz=abs(z['state_pred']-tr['traj'][idx]); return np.r_[dy.mean(),dz.mean(),dz.mean(0).max(),dy.max(),dz.max()]
fixed=np.arange(52,72)
groups={'prescribed':[fixed], 'blocked':[np.arange(i,i+20) for i in (0,13,26,39,52)], 'random':[np.random.default_rng(i).permutation(n)[:20] for i in range(30)], 'coordinate_tails':[np.argsort(sign*x[:,j])[-20:] for j in range(4) for sign in (-1,1)],'radial_tail':[np.argsort(np.linalg.norm(x,axis=1))[-20:]]}
for k,g in groups.items():
    metrics=np.max([evaluate(idx) for idx in g],axis=0); print(k,metrics); assert max(metrics[:3])<.038
for key in ('panel_a','panel_b'):
    v=np.max([evaluate(fixed,key,i) for i in range(len(s[key]['u']))],axis=0); print(key+' jackknife',v); assert max(v[:3])<.038
