import sys, pickle, importlib.util
import numpy as np
path='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('batch_solver',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; original=s['train']; x=original['x']; n=len(x); H=np.c_[original['y'],original['traj']]
def diag(h): return np.array([h[:,0].mean(),h[:,0].std(),h[:,-1].mean(),h[:,-1].std()])
def audit(o):
 assert set(o)=={'y_hat','state_pred','diagnostics_vec'}
 assert all(isinstance(v,np.ndarray) and np.isfinite(v).all() for v in o.values())
 y,T=o['y_hat'],o['state_pred']; assert y.ndim==1 and T.shape==(len(y),len(s['eval_times'])) and o['diagnostics_vec'].shape==(4,)
 np.testing.assert_allclose(o['diagnostics_vec'],diag(np.c_[y,T]),rtol=0,atol=1e-14)
 return np.c_[y,T]
panel=s['panel_c']; u=panel['u']; v=panel['v1']; pairs=((0,1),(0,3),(1,2),(2,3)); affine=mod._affine_design(u); bilinear=np.column_stack([np.ones(len(u)),*[u[:,i]*u[:,j] for i,j in pairs]])
for name,d in [('affine',affine),('four-product',bilinear)]:
 inv=np.linalg.pinv(d); err=(v-d@inv@v)/(1-(d*inv.T).sum(axis=1))[:,None]; print('PANEL v1 LOO',name,'MAE',abs(err).mean(),'max channel MAE',abs(err).mean(axis=0).max(),'median row L2',np.median(np.linalg.norm(err,axis=1)))
cal=mod._calibrate(s['panel_a'],x); b=s['panel_b']; op=np.linalg.lstsq(mod._affine_design(b['u']),b['v'],rcond=None)[0]; mixed=mod._affine_design(cal)@op; D=mod._design(mixed)
rng=np.random.default_rng(159); block={'fixed':[np.arange(n)],'random52/20':[rng.permutation(n) for _ in range(100)],'tail52/20':[]}; xc=(mixed-mixed.mean(0))/mixed.std(0); radii=np.linalg.norm(xc,axis=1)
for v in [radii,*xc.T,*(-xc.T)]: block['tail52/20'].append(np.argsort(v))
for label,splits in block.items():
 scores={key:[] for key in ('artifact','rank2','rank2_oraclefit','quadratic','cubic')}
 from itertools import combinations_with_replacement as cwr
 polynomial=lambda degree:np.column_stack([np.ones(n)]+[np.prod(mixed[:,c],axis=1) for k in range(1,degree+1) for c in cwr(range(4),k)])
 for order in splits:
  tr,te=order[:52],order[52:]; h=H[te]; public={**s,'train':{**original,**{key:val[tr] for key,val in original.items() if key!='summary'}},'test':{'x':x[te]}}
  preds={'artifact':audit(mod.solve_batch(public))}
  cp=preds['artifact']; _,_,V=np.linalg.svd(H[tr,1:],full_matrices=False); V=V[:2]; preds['rank2']=np.c_[cp[:,0],cp[:,1:]@V.T@V];
  f=D[tr]@np.linalg.lstsq(D[tr],H[tr,1:],rcond=None)[0]; _,_,V=np.linalg.svd(f,full_matrices=False); V=V[:2]; preds['rank2_oraclefit']=np.c_[cp[:,0],cp[:,1:]@V.T@V]
  for name,degree in [('quadratic',2),('cubic',3)]:
   d=polynomial(degree); preds[name]=d[te]@np.linalg.lstsq(d[tr],H[tr],rcond=None)[0]
  for key,p in preds.items():
   e=abs(p-h);scores[key].append(np.r_[e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max(),abs(diag(h)-diag(p))])
 for key,q in scores.items(): print(label,key,'avg',np.round(np.mean(q,0),6),'worst',np.round(np.max(q,0),6))
 if label=='fixed': assert max(scores['artifact'][0][:3])<=.038
p=audit(mod.solve_batch(s)); print('FULL',p.shape,diag(p)); assert p.shape==(len(s['test']['x']),1+len(s['eval_times']))
