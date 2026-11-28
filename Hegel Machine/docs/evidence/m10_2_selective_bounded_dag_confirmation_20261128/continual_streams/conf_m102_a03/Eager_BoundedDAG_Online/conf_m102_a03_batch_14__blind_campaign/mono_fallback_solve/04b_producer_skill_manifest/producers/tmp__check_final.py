import pickle,numpy as np,time
import challenger as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train'];
for i,j in [(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(812).permutation(72),3)]:
 d=dict(s,train={k:(a[i] if k!='summary' else a) for k,a in t.items()},test={'x':t['x'][j]});
 for model in [m.solve_batch,m.candidate]:
  o=model(d); r=abs(o['state_pred']-t['traj'][j]);print(model.__name__,len(i),np.mean(abs(o['y_hat']-t['y'][j])),r.mean(),r.mean(0).max(),flush=True)
