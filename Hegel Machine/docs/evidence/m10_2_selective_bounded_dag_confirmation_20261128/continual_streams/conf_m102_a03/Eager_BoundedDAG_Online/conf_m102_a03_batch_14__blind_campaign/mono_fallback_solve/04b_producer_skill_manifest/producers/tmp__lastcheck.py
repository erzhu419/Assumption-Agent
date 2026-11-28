import numpy as np,pickle,json
import challenger as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; modes=['solve_batch','candidate','multiscale','hybrid','hybrid_baseline']; results={k:[] for k in modes}
I=np.arange(len(t['y'])); folds=[(I[:52],I[52:])]+[(np.setdiff1d(I,j),j) for j in np.array_split(np.random.default_rng(812).permutation(len(I)),3)]
for i,j in folds:
 d=dict(s,train={k:(a[i] if k!='summary' else a) for k,a in t.items()},test={'x':t['x'][j]}); a,b,c=[getattr(m,k)(d) for k in modes[:3]]; h={'y_hat':.65*c['y_hat']+.35*b['y_hat'],'state_pred':.35*c['state_pred']+.65*b['state_pred']}; hb={'y_hat':.35*c['y_hat']+.25*b['y_hat']+.40*a['y_hat'],'state_pred':h['state_pred']}
 for k,o in zip(modes,[a,b,c,h,hb]):
  e=np.abs(o['y_hat']-t['y'][j]); f=np.abs(o['state_pred']-t['traj'][j]); metrics=(float(e.mean()),float(f.mean()),float(f.mean(0).max()));results[k].append(metrics);print(k,len(i),*metrics,flush=True)
json.dump(results,open('/tmp/lastmetrics.json','w'))
