import copy,importlib.util,pickle,numpy as np, shutil
base='/holdout_task/batch_solver.py'
def load(path):
 s=importlib.util.spec_from_file_location('candidate',path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
study=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
train=study['train']; n=len(train['y']); rng=np.random.default_rng(295)
splits=[np.arange(n)]+[rng.permutation(n) for _ in range(3)]
def checked(m,st):
 p=m.solve_batch(st); y=p['y_hat']; z=p['state_pred']; d=p['diagnostics_vec']; L=len(st['test']['x']); assert isinstance(y,np.ndarray) and isinstance(z,np.ndarray) and isinstance(d,np.ndarray); assert y.shape==(L,) and z.shape==(L,len(st['eval_times'])) and d.shape==(4,); assert all(np.isfinite(q).all() for q in (y,z,d)); np.testing.assert_allclose(d,[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()],atol=1e-12); return y,z
base_scores=None
for path in [base,'/tmp/b06_adaptive_candidate.py']:
 try:
  m=load(path); checked(m,copy.deepcopy(study)); result=[]
  for ii in splits:
   i,j=ii[:52],ii[52:]; st=copy.deepcopy(study); st['train']={k:train[k][i].copy() for k in ('x','y','traj')}; y,z=st['train']['y'],st['train']['traj']; st['train']['summary']=np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]); st['test']={'x':train['x'][j].copy()}; y,z=checked(m,st); hourly=np.mean(abs(z-train['traj'][j]),0); result.append([np.mean(abs(y-train['y'][j])),hourly.mean(),hourly.max()]);
  e=np.array(result);print(path,'rows: prescribed then random 52/20; columns: y, trajectory, worst time',e.tolist(),flush=True)
  if path==base:base_scores=e
  elif base_scores is None or (np.all(e[0,:2]<=.038) or (np.all(e[0,:2]<base_scores[0,:2]) and e.max()<=1.05*base_scores.max())):
   shutil.copyfile(path,base);print('Promoted adaptive candidate',flush=True)
 except Exception as ex:
  print(path,'candidate check failed:',repr(ex),flush=True)
