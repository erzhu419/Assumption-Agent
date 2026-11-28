import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; rng=np.random.default_rng(405); pi=np.arange(72)
for label,p in [('fixed',pi),('shuffled',rng.permutation(72)),('reversed',pi[::-1]),('large-x',np.argsort(np.linalg.norm(tr['x'],axis=1)))]:
    a,b=p[:52],p[52:]; d={**s,'train':{**tr,**{k:tr[k][a] for k in ['x','y','traj']}},'test':{'x':tr['x'][b]}}
    y,t,ey,et=m._candidate_predictions(d); print(label,'scalar',abs(y-tr['y'][b]).mean(1),'trajectory',abs(t-tr['traj'][b]).mean((1,2)), 'LOO',ey,et,flush=True)
    if label=='fixed':
        o=m.solve_batch(d); err=abs(o['state_pred']-tr['traj'][b]); print('fixed blend',abs(o['y_hat']-tr['y'][b]).mean(),err.mean(), 'worst time',err.mean(0).max(),flush=True)
