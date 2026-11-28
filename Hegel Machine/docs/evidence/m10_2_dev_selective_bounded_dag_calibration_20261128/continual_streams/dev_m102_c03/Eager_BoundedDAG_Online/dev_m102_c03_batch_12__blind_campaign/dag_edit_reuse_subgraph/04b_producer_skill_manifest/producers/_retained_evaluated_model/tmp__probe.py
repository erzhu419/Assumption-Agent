import pickle, numpy as np, importlib.util
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.linalg.norm(v,axis=1)<5
print('A kept',keep.sum())
for j in range(4):
 print('channel',j)
 for deg in (1,2,3,4):
  ids=[()] + [q for d in range(1,deg+1) for q in combinations_with_replacement(range(4),d)]; F=np.column_stack([np.prod(u[:,q],axis=1) for q in ids]); b=np.linalg.lstsq(F[keep],v[keep,j],rcond=None)[0]; print(deg,np.std((F@b-v[:,j])[keep]),[(q,round(c,4)) for q,c in zip(ids,b) if abs(c)>.025][:24])
for f in ('00','10'):
 spec=importlib.util.spec_from_file_location('arch'+f,'/tmp/hegel_continual_archive/dev_m102_c03_batch_'+f+'_solver.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 tr=s['train']; out=mod.solve_batch(dict(s,train={k:v[:52] if k!='summary' else v for k,v in tr.items()},test={'x':tr['x'][52:]}));print('arch',f, np.mean(abs(out['y_hat']-tr['y'][52:])),np.mean(abs(out['state_pred']-tr['traj'][52:])))
 if f=='10':
  p=s['panel_c']; z=p['u']; w=mod._panels(s,z); print('arch C residual',np.sqrt(np.mean((w[:,:4]-p['v1'])**2)),np.sqrt(np.mean((w[:,4:]-p['v2'])**2)))
