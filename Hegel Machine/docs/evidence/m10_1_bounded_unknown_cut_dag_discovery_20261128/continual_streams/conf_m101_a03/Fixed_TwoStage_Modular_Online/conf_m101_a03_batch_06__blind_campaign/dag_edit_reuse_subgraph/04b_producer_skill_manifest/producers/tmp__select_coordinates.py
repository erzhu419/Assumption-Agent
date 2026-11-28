import pickle,runpy,numpy as np
from pathlib import Path
base=Path('/tmp/node_previous.py').read_text().split('\ndef _coordinate_model(',1)[0]
new=Path('/tmp/coordinate_search.py').read_text()
mod={};exec(base+'\n'+new,mod)
sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=st['train']; n=len(tr['x']); limit=max(40,n-20)
a=mod['_fit_panel_a'](st['panel_a']); b=mod['_fit_panel_b'](st['panel_b'])
q=mod['_inverse_a'](tr['x'][:limit],a); v=q@b[1:]+b[0]; y=tr['y'][:limit]; t=tr['traj'][:limit]; i,j=np.triu_indices(4); Q=q[:,i]*q[:,j]
folds=np.random.default_rng(647).permutation(limit).reshape(-1); folds=np.array_split(folds,3)
configs=[(0,0.,0.,1.),(1,.14,1.1,2.),(1,.055,.45,1.),(1,.24,.55,.6),(2,.14,.85,2.),(2,.07,.35,.7),(1,.09,2.5,5.)]
results=[]
for cfg in configs:
    errors=[]
    for te in folds:
        ti=np.setdiff1d(np.arange(limit),te)
        M,E=mod['_handoff'](q[ti],v[ti],y[ti],t[ti],cfg)
        z=v@M+Q@E
        pred=sink({'node_00':{'z':z[ti],'z_test':z[te],'y':y[ti],'traj':t[ti]}})
        errors.append((np.mean(abs(np.asarray(pred['y_hat']).reshape(-1)-y[te].reshape(-1))),np.mean(abs(pred['state_pred']-t[te]))))
    errs=np.asarray(errors); print(cfg,np.round(errs,6).tolist(),flush=True)
    results.append((.65*errs[:,0].mean()+.35*errs[:,0].max()+max(0.,errs[:,1].max()-.035),cfg))
    best=min(results)[1]
    Path('/holdout_task/node_00.py').write_text(base+'\n'+new.replace('_CONFIGURATION = (1,.14,1.1,2.)','_CONFIGURATION = '+repr(best)))
print('Selected',min(results)[1],flush=True)
