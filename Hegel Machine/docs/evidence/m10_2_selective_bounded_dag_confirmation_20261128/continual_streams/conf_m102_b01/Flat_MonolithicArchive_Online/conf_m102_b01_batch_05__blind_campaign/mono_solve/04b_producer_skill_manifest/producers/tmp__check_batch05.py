import importlib.util
import pickle
import numpy as np
path = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', path+'batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(path+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; idx=np.arange(len(tr['x']))
def test(i,j,study=s):
    p=m.solve_batch(dict(study, train=dict(x=tr['x'][i], y=tr['y'][i], traj=tr['traj'][i]), test=dict(x=tr['x'][j])))
    ey=np.abs(p['y_hat']-tr['y'][j]); et=np.abs(p['state_pred']-tr['traj'][j]); return ey.mean(), et.mean(), et.mean(0).max(), ey.max(), et.max()
print('Prescribed 52/20:', test(idx[:52],idx[52:]))
rng=np.random.default_rng(617)
runs=[]
for k in range(30):
    i=rng.permutation(idx); runs.append(test(i[:52],i[52:]))
for z in [m._latent(s,tr['x']),tr['x']]:
    for proj in (*z.T, np.linalg.norm(z-z.mean(0),axis=1)):
        i=idx[np.argsort(proj)]; runs.extend([test(i[:52],i[52:]), test(i[20:],i[:20])])
print('Worst random or directional holdout:',np.max(runs,axis=0))
for key in ('panel_a','panel_b','panel_c'):
    panel=s[key]; runs=[]
    for k in range(len(panel['u'])):
        q=dict(s);q[key]={name:np.delete(val,k,axis=0) for name,val in panel.items()}
        runs.append(test(idx[:52],idx[52:],q))
    print('Worst leave-one-row-out:',key,np.max(runs,axis=0))
p=m.solve_batch(s);print('Full test diagnostics:',p['diagnostics_vec'])
