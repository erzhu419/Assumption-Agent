import pickle, runpy, numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/batch_solver.py'); sol=m['solve_batch']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; allrows=np.arange(len(x))
splits=[('52/20',allrows[:52],allrows[52:]),('56/16',allrows[:56],allrows[56:])]
for i,te in enumerate(np.array_split(np.random.default_rng(35).permutation(52),4)):
 splits.append(('inner'+str(i),np.setdiff1d(allrows[:52],te),te))
for name,tr,te in splits:
 out=sol(dict(s,train=dict(s['train'],x=x[tr],y=y[tr],traj=t[tr]),test={'x':x[te]})); e=np.abs(out['y_hat']-y[te]); et=np.abs(out['state_pred']-t[te]); print(name, 'MAE',e.mean(),et.mean(),'per-time',np.round(et.mean(0),5),'max row',te[e.argmax()],e.max())
 if name=='52/20': assert e.mean() <= .038
