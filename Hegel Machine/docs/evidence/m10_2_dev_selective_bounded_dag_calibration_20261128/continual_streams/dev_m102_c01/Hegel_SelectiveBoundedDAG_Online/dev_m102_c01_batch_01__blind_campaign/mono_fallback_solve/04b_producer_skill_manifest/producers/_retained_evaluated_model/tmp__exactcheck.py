import sys, pickle, importlib.util, numpy as np
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=st['train']; x=tr['x']; n=len(x); ids=np.arange(n)
splits=[(ids[:52],ids[52:])]+[(j[:52],j[52:]) for j in (np.random.default_rng(i).permutation(n) for i in range(32))]+[(np.setdiff1d(ids,j),j) for j in np.array_split(np.argsort(np.linalg.norm(x,axis=1)),4)]
for p in sys.argv[1:]:
 spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 errs=[]; ts=[]
 for a,b in splits:
  out=m.solve_batch(dict(st,train={k:v[a] for k,v in tr.items() if k!='summary'},test={'x':x[b]})); dt=np.abs(out['state_pred']-tr['traj'][b]); ts.append(dt.mean(0)); errs.append([np.mean(np.abs(out['y_hat']-tr['y'][b])),dt.mean(),dt[:,-1].mean()])
 e=np.array(errs); print(p, 'prescribed',e[0], 'random mean',e[1:33].mean(0),'random worst',e[1:33].max(0),'radial worst',e[33:].max(0),'prescribed per-time',ts[0])
