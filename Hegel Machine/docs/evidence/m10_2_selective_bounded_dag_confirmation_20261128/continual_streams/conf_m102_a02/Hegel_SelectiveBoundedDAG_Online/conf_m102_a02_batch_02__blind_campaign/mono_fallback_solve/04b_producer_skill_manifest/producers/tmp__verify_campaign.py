import importlib.util
import pickle
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
path='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('campaign_solver',path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t=s['train']; x=t['x']; ids=np.arange(len(x)); y=t['y']; traj=t['traj']
def validate(o,n):
    assert {k:a.shape for k,a in o.items()}=={'y_hat':(n,),'state_pred':(n,8),'diagnostics_vec':(4,)}
    assert all(np.isfinite(a).all() for a in o.values())
    u=o['y_hat']; v=o['state_pred'][:,-1]
    np.testing.assert_array_equal(o['diagnostics_vec'],[u.mean(),u.std(),v.mean(),v.std()])
def split(a,b):
    study={**s,'train':{**t,**{k:t[k][a] for k in ('x','y','traj')}},'test':{'x':x[b]}}
    o=m.solve_batch(study); validate(o,len(b))
    return np.mean(abs(o['y_hat']-y[b])),np.mean(abs(o['state_pred']-traj[b]),axis=0)
first,state=split(ids[:52],ids[52:]); print('52/20 scalar:',first,'trajectory-by-time:',state)
assert first<=.038
for label,folds in [('4-fold seed 12',KFold(4,shuffle=True,random_state=12).split(x)),('4-fold seed 73',KFold(4,shuffle=True,random_state=73).split(x)),('fresh 52/20',ShuffleSplit(96,test_size=20,random_state=1199).split(x))]:
    scores=[split(a,b) for a,b in folds]; w=max(a for a,_ in scores); tw=np.max([b for _,b in scores]); print(label,'worst scalar:',w,'worst time MAE:',tw); assert w<=.038
z=(x-m._calibration(s['panel_a'])[0])/m._calibration(s['panel_a'])[1]
for name,rank in [('gate boundary',abs(z[:,2])),('large norm',-np.linalg.norm(z,axis=1))]:
    b=np.argsort(rank)[:20]; a=np.setdiff1d(ids,b); sc,st=split(a,b); print(name,'scalar:',sc,'worst time MAE:',max(st))
    assert sc <= .038
validate(m.solve_batch(s),len(s['test']['x']))
