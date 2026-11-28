import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=m._coordinates(s['panel_a'],tr['x']); rng=np.random.default_rng(9026)
slices=[('52/20',np.arange(52),np.arange(52,72)),('56/16',np.arange(56),np.arange(56,72))]; slices += [('random'+str(k),p[:56],p[56:]) for k,p in enumerate([rng.permutation(72) for _ in range(12)])]
for name,val in [('norm',np.linalg.norm(z,axis=1)),*[(f'{j}{sign:+}',sign*z[:,j]) for j in range(4) for sign in (-1,1)]]:
 p=np.argsort(val); slices.append((name,p[:-16],p[-16:]))
for name,i,j in slices:
 ss=dict(s,train={k:tr[k][i] for k in ('x','y','traj')},test={'x':tr['x'][j]}); out=m.solve_batch(ss); ye=np.abs(out['y_hat']-tr['y'][j]); se=np.abs(out['state_pred']-tr['traj'][j]); print(name,*(f'{x:.5f}' for x in (ye.mean(),se.mean(),se[:,-1].mean())))
