import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task'); import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=map(s['panel_a'].get,['u','v']); b,w=m._calibration(s['panel_a']); keep=np.linalg.norm(u@w+b-v,axis=1)<2; u,v=u[keep],v[keep]
np.set_printoptions(precision=4,suppress=True)
print('ranges',u.min(0),u.max(0),v.min(0),v.max(0),s['train']['x'].min(0),s['train']['x'].max(0)); print('pairs',np.c_[u,v][:12]);
for d in [1,2,3,4]:
 terms=[p for k in range(d+1) for p in itertools.combinations_with_replacement(range(4),k)]; a=np.column_stack([np.prod(u[:,p],axis=1) if p else np.ones(len(u)) for p in terms]); pinv=np.linalg.pinv(a); coef=pinv@v; r=v-a@coef; print('degree',d,'rmse',np.sqrt(np.mean(r*r)),'oof',np.sqrt(np.mean((r/np.maximum(1-np.sum(a*pinv.T,axis=1),1e-10)[:,None])**2))); 
 if d<=3:
  for p,c in zip(terms,coef):print(p,c)
for name,h in [('abs',u*abs(u)),('cube',u**3),('square',u**2),('tanh',np.tanh(2.15*u)),('sin',np.sin(2.15*u)),('rational',u/(1+2.15*u*u)),('absrat',u/(1+2.15*abs(u))),('square_rat',u*u/(1+2.15*u*u))]:
 a=np.c_[np.ones(len(u)),u,h]; pinv=np.linalg.pinv(a); coef=pinv@v; r=v-a@coef; print(name,np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/(1-np.sum(a*pinv.T,axis=1))[:,None])**2)))
