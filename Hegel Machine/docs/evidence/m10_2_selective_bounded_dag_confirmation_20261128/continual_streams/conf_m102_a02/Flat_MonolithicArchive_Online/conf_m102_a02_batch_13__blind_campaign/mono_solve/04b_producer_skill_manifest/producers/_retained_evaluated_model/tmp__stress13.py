import numpy as np, pickle, runpy
m = runpy.run_path('/holdout_task/batch_solver.py'); st = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr = st['train']; z = m['_operator'](st['panel_b'])(m['_inverse'](tr['x'], m['_calibration'](st['panel_a']))); a = m['_affine'](z); ls = m['_ls']; S = m['_scalar']; y = tr['y']; t = tr['traj']; ix = np.arange(len(y));
folds = [('52/20',ix[:52],ix[52:])]
for seed in [41,73,19]:
    for k,j in enumerate(np.array_split(np.random.default_rng(seed).permutation(len(y)),4)):
        folds.append((str(seed)+'/'+str(k),np.setdiff1d(ix,j),j))
for k in range(4):
    order = np.argsort(z[:,k]);
    for tag,j in [('low',order[:14]),('high',order[-14:])]:
        folds.append((f'latent-{tag}-{k}',np.setdiff1d(ix,j),j))
for name,i,j in folds:
    ps = [S(z[i],z[j],y[i],mode=k) for k in ('full','sparse','rational')]; ps += [(ps[0]+ps[1])/2,(ps[0]+ps[2])/2]; te = abs(a[j]@ls(a[i],t[i])-t[j]); print(name,np.round([np.mean(abs(p-y[j])) for p in ps],5),'state mean/max time',round(te.mean(),5),round(te.mean(0).max(),5),flush=True)
out = m['solve_batch'](st); assert all(isinstance(v,np.ndarray) and np.isfinite(v).all() for v in out.values()); yq=out['y_hat']; last=out['state_pred'][:,-1]; np.testing.assert_array_equal(out['diagnostics_vec'],[yq.mean(),yq.std(),last.mean(),last.std()]); print('All output types and diagnostics verified.')
