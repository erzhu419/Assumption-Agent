import copy,importlib.util,pickle,sys,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']
p=importlib.util.spec_from_file_location('cand',sys.argv[1] if len(sys.argv)>1 else '/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
for name,h in [('public',np.arange(52,72)),('first',np.arange(20)),('middle',np.arange(24,44)),('corners',np.argsort(np.linalg.norm(x,axis=1))[-20:]),('x0-low',np.argsort(x[:,0])[:20]),('x3-high',np.argsort(x[:,3])[-20:])]:
 k=np.setdiff1d(np.arange(len(x)),h); t=copy.deepcopy(s); t['train']={key:t['train'][key][k] for key in ('x','y','traj')}; t['test']={'x':x[h]};r=m.solve_batch(t); e=np.abs(r['y_hat']-s['train']['y'][h]);f=np.abs(r['state_pred']-s['train']['traj'][h]);print(name,round(e.mean(),6),round(f.mean(),6),'worst horizon',round(f.mean(0).max(),6),flush=True)
