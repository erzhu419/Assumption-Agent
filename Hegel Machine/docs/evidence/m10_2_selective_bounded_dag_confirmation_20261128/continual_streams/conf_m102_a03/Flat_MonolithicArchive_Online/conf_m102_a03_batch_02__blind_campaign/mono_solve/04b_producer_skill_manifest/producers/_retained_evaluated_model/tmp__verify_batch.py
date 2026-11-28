import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; ids=np.arange(len(x)); z=m._decode(x,m._calibration(s['panel_a']))
folds={'specified_52_20':[(ids[:52],ids[52:])], 'alternate_54_18':[], 'fresh_54_18':[], 'coordinate_tails':[]}
for key,seeds in [('alternate_54_18',[29,487,8701]),('fresh_54_18',[137,991,1709])]:
 for seed in seeds:
  for j in np.array_split(np.random.default_rng(seed).permutation(len(ids)),4): folds[key].append((np.setdiff1d(ids,j),j))
for k in range(z.shape[1]):
 a=np.argsort(z[:,k]);
 for j in (a[:8],a[-8:]): folds['coordinate_tails'].append((np.setdiff1d(ids,j),j))
for name,fs in folds.items():
 e=[]; st=[]
 for i,j in fs:
  train={**s['train'],'x':x[i], 'y':y[i], 'traj':t[i], 'summary':np.array([y[i].mean(),y[i].std(),t[i,-1].mean(),t[i,-1].std()])}
  out=m.solve_batch({**s,'train':train,'test':{'x':x[j]}}); p=out['y_hat']; u=out['state_pred']; e.append(np.abs(p-y[j]).mean());st.append(np.abs(u-t[j]).mean(axis=0));np.testing.assert_allclose(out['diagnostics_vec'],[p.mean(),p.std(),u[:,-1].mean(),u[:,-1].std()]); assert np.isfinite(u).all() and np.isfinite(p).all()
 print(name,'y mean/worst',np.mean(e),max(e),'traj mean',np.mean(st),'worst horizon',np.max(st),'terminal worst',np.max(np.asarray(st)[:,-1]));
 if name=='specified_52_20': assert e[0]<=.038
