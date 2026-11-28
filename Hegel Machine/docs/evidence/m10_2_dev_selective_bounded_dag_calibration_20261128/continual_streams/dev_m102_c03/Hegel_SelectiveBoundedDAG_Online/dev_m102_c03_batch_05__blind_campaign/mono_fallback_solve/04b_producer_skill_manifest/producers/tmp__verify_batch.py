import importlib.util,pickle,numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train'];N=len(tr['x']); rng=np.random.default_rng(2026)
c,d=m._branches(s,tr['x']);hy=m._one(np.c_[c,d,np.sum(c*d,axis=1)]);ht=np.c_[hy,c[:,0]*d[:,1]]; w,q=m._fit_state(ht[:52],s['eval_times'],tr['traj'][:52]); print('weights y:',m._fit(hy[:52],tr['y'][:52]).round(5));print('weights state:',w.round(5));print('q:',q.round(5));
splits=[('prescribed',np.arange(52,N)),('earliest',np.arange(20))]+[(f'random{i}',rng.choice(N,20,replace=False)) for i in range(24)];
for val,name in [(np.linalg.norm(tr['x'],axis=1),'norm')]+[(tr['x'][:,i],f'x{i}') for i in range(4)]:
 order=np.argsort(val);splits.extend([(name+' low',order[:20]),(name+' high',order[-20:])]);
values=[]
for name,ii in splits:
 jj=np.setdiff1d(np.arange(N),ii);st=dict(s,train={**tr,**{k:tr[k][jj] for k in ('x','y','traj')}},test={'x':tr['x'][ii]});o=m.solve_batch(st);ye=np.abs(o['y_hat']-tr['y'][ii]);te=np.abs(o['state_pred']-tr['traj'][ii]); row=(ye.mean(),te.mean(),te.mean(0).max(),ye.max(),te.max());values.append(row);print(name,np.round(row,6));
 assert max(row[:3]) < .038
print('max across splits',np.max(values,axis=0));o=m.solve_batch(s);print('real test ranges', o['y_hat'].min(),o['y_hat'].max(),o['state_pred'].min(),o['state_pred'].max()); np.testing.assert_allclose(o['diagnostics_vec'],[o['y_hat'].mean(),o['y_hat'].std(),o['state_pred'][:,-1].mean(),o['state_pred'][:,-1].std()]);
