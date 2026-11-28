import importlib.util
import pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; rng=np.random.default_rng(29)
def score(i,j):
    st=dict(s,train={**tr, **{k:tr[k][i] for k in ('x','y','traj')}},test={'x':tr['x'][j]})
    pred=m.solve_batch(st)
    dy=abs(pred['y_hat']-tr['y'][j]); dt=abs(pred['state_pred']-tr['traj'][j]); last=pred['state_pred'][:,-1]
    np.testing.assert_allclose(pred['diagnostics_vec'],[pred['y_hat'].mean(),pred['y_hat'].std(),last.mean(),last.std()])
    return dy.mean(),dt.mean(),max(dt.mean(0)),dy.max(),dt.mean(1).max()
base=score(np.arange(52),np.arange(52,72)); print('52/20: scalar, trajectory, worst timestamp, worst scalar row, worst trajectory row',base)
folds=[]
for _ in range(18):
    p=rng.permutation(72);folds.append(score(p[:52],p[52:]))
for v in [*tr['x'].T,np.linalg.norm(tr['x'],axis=1)]:
    p=np.argsort(v);folds.extend([score(p[:52],p[52:]),score(p[20:],p[:20])])
worst=np.max([base,*folds],axis=0); print('worst across predefined and stress splits',worst); assert max(worst[:3]) < .038
out=m.solve_batch(s); assert out['y_hat'].shape==(36,) and out['state_pred'].shape==(36,8) and all(np.isfinite(a).all() for a in out.values())
