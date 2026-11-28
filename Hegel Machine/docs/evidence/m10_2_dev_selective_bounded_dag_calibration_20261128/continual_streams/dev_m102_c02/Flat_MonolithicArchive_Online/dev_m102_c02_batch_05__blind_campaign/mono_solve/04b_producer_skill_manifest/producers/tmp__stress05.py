import importlib.util,pickle,numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x,y,T=t['x'],t['y'],t['traj']; rng=np.random.default_rng(320)
b,g=m._calibrate(s['panel_a']); z=(x-b)/g; d=m._panel_operator(s['panel_b'],z)
N=len(x); ii=np.arange(N); folds=[('public',ii[:52],ii[52:])]+[('random',p[:52],p[52:]) for p in [rng.permutation(N) for _ in range(50)]]
for v in [np.max(np.abs(z),axis=1),*z.T,*d.T]:
 p=np.argsort(v);folds.extend([('tail',p[:52],p[52:]),('tail',p[20:],p[:20])])
for name in ['public','random','tail']:
 es=[]; h=[]; term=[]
 for k,i,j in folds:
  if k!=name:continue
  sub=dict(s,train={q:t[q][i] for q in ('x','y','traj')},test={'x':x[j]});o=m.solve_batch(sub); ay=np.abs(o['y_hat']-y[j]); at=np.abs(o['state_pred']-T[j]); es.append([ay.mean(),at.mean()]); h.append(at.mean(0)); term.append(at[:,-1].mean()); assert np.allclose(o['diagnostics_vec'],[o['y_hat'].mean(),o['y_hat'].std(),o['state_pred'][:,-1].mean(),o['state_pred'][:,-1].std()]);
 print(name,len(es),'mean',np.mean(es,0),'worst',np.max(es,0),'worst hours',np.max(h,0))
 assert np.max(es)<.038
