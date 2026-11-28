import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,t=(tr[k] for k in ('x','y','traj')); allidx=np.arange(len(x)); z=np.linalg.norm(x,axis=1)
rng=np.random.default_rng(517)
splits={'public':allidx[52:], 'radius':np.argsort(z)[-20:], 'amplitude':np.argsort(abs(t[:,-1]))[-20:]};
for j in range(4):
 for sign in [-1,1]: splits[f'edge{j},{sign}']=np.argsort(x[:,j]*sign)[-20:]
for j in range(15): splits[f'random{j}']=rng.permutation(len(x))[:20]
for v in ['joint','linear','separate']:
 res={}
 for label,j in splits.items():
  i=np.setdiff1d(allidx,j); d={**s,'train':{'x':x[i],'y':y[i],'traj':t[i]},'test':{'x':x[j]}}; a,b=m._predict(d,v); e=abs(b-t[j]); res[label]=np.array([abs(a-y[j]).mean(),e.mean(),e.mean(axis=0).max(),e[:,-1].mean()])
 print(v,'public',res['public'],'all maxima',np.max(list(res.values()),axis=0),'worst slices',[(max(res,key=lambda s:res[s][j]),round(max(r[j] for r in res.values()),6)) for j in range(4)]); print('random mean',np.mean([row for name,row in res.items() if name.startswith('random')],axis=0))
