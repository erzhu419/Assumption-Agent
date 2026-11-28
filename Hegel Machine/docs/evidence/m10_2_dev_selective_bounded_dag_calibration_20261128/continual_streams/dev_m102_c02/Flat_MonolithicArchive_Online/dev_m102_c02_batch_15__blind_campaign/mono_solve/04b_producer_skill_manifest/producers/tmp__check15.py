import importlib.util, pickle, numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; rng=np.random.default_rng(6341)
splits=[np.arange(52,72),np.arange(20),np.arange(24,44),np.argsort(np.linalg.norm(x,axis=1))[-20:]]+[rng.choice(72,20,replace=False) for _ in range(6)]
worst=np.zeros(3)
for i,te in enumerate(splits):
 a=np.setdiff1d(np.arange(72),te); st={**s,'train':{k:(v[a] if k!='summary' else v) for k,v in tr.items()},'test':{'x':x[te]}}
 o=m.solve_batch(st); ey=np.abs(o['y_hat']-tr['y'][te]); et=np.abs(o['state_pred']-tr['traj'][te]); score=np.array([ey.mean(),et.mean(),et.mean(0).max()]); worst=np.maximum(worst,score)
 print(i, 'y, trajectory, worst horizon',score,'max individual',ey.max(),et.max()); assert np.all(score < .038)
print('worst',worst)
