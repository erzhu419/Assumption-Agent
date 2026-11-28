import importlib.util,pickle,numpy as np,shutil
p='/holdout_task/batch_solver.py'; sp=importlib.util.spec_from_file_location('model',p); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ids=np.arange(72); rng=np.random.default_rng(142); cuts=[np.arange(52,72),np.arange(20),rng.choice(ids,20,replace=False)]; weights=[0.,.25,.5,.75,1.]; metrics=[]
for te in cuts:
 a=np.setdiff1d(ids,te); sub=dict(s,train={k:tr[k][a] for k in ['x','y','traj']},test={'x':tr['x'][te]}); old=m._v02(sub); y,t=m._algebraic(sub)
 e=np.array([[np.mean(abs((1-w)*old['y_hat']+w*y-tr['y'][te])),np.mean(abs((1-w)*old['state_pred']+w*t-tr['traj'][te]))] for w in weights]); metrics.append(e);print('split',np.round(e,6).tolist(),flush=True)
M=np.array(metrics); chosen=[]
for j in range(2):
 best=min(range(len(weights)),key=lambda k:(.6*M[:,k,j].max()+.4*M[:,k,j].mean(),k) if M[0,k,j]<=M[0,0,j] else (float('inf'),k));chosen.append(weights[best])
if any(chosen):
 with open(p,'a') as f:f.write('\n\ndef solve_batch(study):\n    return _blend(study, '+repr(tuple(chosen))+')\n')
else:shutil.copyfile('/tmp/v02_solver.py',p)
print('Retained blend:',chosen,flush=True)
