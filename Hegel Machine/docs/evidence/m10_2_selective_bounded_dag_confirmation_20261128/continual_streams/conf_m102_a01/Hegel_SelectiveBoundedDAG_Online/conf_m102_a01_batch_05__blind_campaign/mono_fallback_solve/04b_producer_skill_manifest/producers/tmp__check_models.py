import sys,pickle,numpy as np,importlib.util
sys.path.insert(0,'/holdout_task');sys.path.insert(0,'/tmp/hegel_continual_archive')
import batch_solver as new,conf_m102_a01_batch_03_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];n=len(t['y']);rng=np.random.default_rng(502)
z=new._calibrate(s['panel_a'],t['x']); radius=np.linalg.norm(z,axis=1);r=np.argsort(radius)
splits=[(np.arange(52),np.arange(52,72))]+[(a[:40],a[40:]) for a in [rng.permutation(52) for _ in range(6)]]+[(r[:52],r[52:]),(r[20:],r[:20])]+[(a[:52],a[52:]) for a in [rng.permutation(72) for _ in range(5)]]
E=[]
for i,(tr,te) in enumerate(splits):
 train={k:v[tr] for k,v in t.items() if k!='summary'}; y=train['y']; end=train['traj'][:,-1];train['summary']=np.array([y.mean(),y.std(),end.mean(),end.std()]);q=dict(s,train=train,test={'x':t['x'][te]});p=new.solve_batch(q);o=old.solve_batch(q)
 def metric(a):
  dy=abs(a['y_hat']-t['y'][te]);dt=abs(a['state_pred']-t['traj'][te]);return np.array([dy.mean(),dt.mean(),dt.mean(0).max()])
 e=[metric(p),metric(o),metric({k:.8*p[k]+.2*o[k] for k in ['y_hat','state_pred']})];E.append(e);print(i,np.round(e,5).tolist())
print('mean',np.mean(E,axis=0),'worst',np.max(E,axis=0))
