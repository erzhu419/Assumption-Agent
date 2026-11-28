import importlib.util,pickle,numpy as np
p='/holdout_task/'
sp=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; rng=np.random.default_rng(451)
splits=[('prescribed',np.arange(52,72)),('early',np.arange(20)),('middle',np.arange(26,46))]
for key,ref in [('x-norm',np.linalg.norm(tr['x'],axis=1)),('largest-y',abs(tr['y'])),('largest-terminal',abs(tr['traj'][:,-1]))]: splits.append((key,np.argsort(ref)[-20:]))
splits+= [(f'random-{j}',rng.permutation(72)[:20]) for j in range(8)]
for name,j in splits:
 i=np.setdiff1d(np.arange(72),j)
 o=m.solve_batch({**s,'train':{**tr,**{k:tr[k][i] for k in ('x','y','traj')}},'test':{'x':tr['x'][j]}})
 er=np.abs(o['state_pred']-tr['traj'][j]); y=np.mean(abs(o['y_hat']-tr['y'][j])); print(name, 'scalar',round(y,7),'traj',round(er.mean(),7),'worst_time',round(er.mean(0).max(),7)); assert max(y,er.mean(),er.mean(0).max())<=.038
print('calibration:',m._calibration(s['panel_a']))
cal=m._calibration(s['panel_a']); h=m._operator(s['panel_b'])(m._inverse(tr['x'],cal)); print('scalar coefficients:',m._ls(m._scalar_design(h),tr['y'])); print('scalar train rms:',np.std(m._scalar_design(h)@m._ls(m._scalar_design(h),tr['y'])-tr['y']))
