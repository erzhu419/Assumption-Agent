import numpy as np,pickle,runpy
from scipy.optimize import least_squares
M=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; t=s['eval_times']; Y=np.c_[d['y'],d['traj']]; w=M['_operator'](s['panel_b'])(M['_decode'](d['x'],M['_calibration'](s['panel_a']))); X=M['_affine'](M['_quadratic'](w)); cols=['1',*['w'+str(i) for i in range(4)],*[f'{i}{j}' for i,j in zip(*np.triu_indices(4))]]; fit=M['_fit']; T=np.arange(52); J=np.arange(52,72)
for ids in [T,np.arange(72)]:
 c=fit(X[ids],Y[ids]); print('coeff',len(ids)); print('name        y     early      last'); print(*[(n,*np.round(f[[0,1,-1]],5)) for n,f in zip(cols,c)],sep='\n'); print('train MAE',abs(X[ids]@c-Y[ids]).mean(0)); print('hold MAE',abs(X[J]@c-Y[J]).mean(0)); print('trajectory coefficient sv',np.linalg.svd(c[:,1:],compute_uv=False))
for g in [np.arange(4),np.arange(4)+1]:
 print(g)
q=dict(s,train={k:v[T] for k,v in d.items() if k!='summary'},test={'x':d['x'][J]}); q['train']['summary']=np.array([d['y'][T].mean(),d['y'][T].std(),d['traj'][T,-1].mean(),d['traj'][T,-1].std()]); r=M['solve_batch'](q); print('baseline y',abs(r['y_hat']-Y[J,0]).mean(),'traj by time',abs(r['state_pred']-Y[J,1:]).mean(0)); print('times',t,'telemetry',s['telemetry_vec'])
