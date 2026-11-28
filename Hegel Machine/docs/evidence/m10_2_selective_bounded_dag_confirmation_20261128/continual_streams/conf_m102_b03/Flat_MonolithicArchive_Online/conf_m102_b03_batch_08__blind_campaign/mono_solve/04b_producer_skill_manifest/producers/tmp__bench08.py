import sys,pickle,numpy as np,importlib.util
spec=importlib.util.spec_from_file_location('base','/tmp/fallback08.py'); b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
exec(open('/tmp/challenger08.py').read(),b.__dict__)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; scores=[]
for te in [np.arange(52,72)]+[np.random.default_rng(k).permutation(72)[:20] for k in range(5)]:
 it=np.setdiff1d(np.arange(72),te); a={**s,'train':{**tr,**{k:tr[k][it] for k in ('x','y','traj')}},'test':{'x':tr['x'][te]}}
 base=b._predict(a); ys,ts=b._challenge(a); es=[]
 for j,cols,P in [(0,slice(0,1),ys),(1,slice(1,None),ts)]:
  es.append([np.mean(abs(f*V+(1-f)*base[:,cols]-Y[te,cols])) for V in P for f in (0,.45,.8,1.)]);
 scores.append(es)
 print('fold',np.round(es[0],4),np.round(es[1],4),flush=True)
ix=[]
for j in (0,1):
 A=np.array([u[j] for u in scores]); score=.5*A[0]+.35*A[1:].mean(0)+.15*A[1:].max(0); i=int(np.argmin(score)); print('selected',j,i,'base',A[:,0],'challenger',A[:,i]); ix.append(i)
if any(i%4 for i in ix):
 with open('/holdout_task/batch_solver.py','a') as f:
  f.write('\n'+open('/tmp/challenger08.py').read()); f.write('\n\ndef solve_batch(study: dict) -> dict:\n    base=_predict(study)\n    ys,ts=_challenge(study)\n')
  for j,i in enumerate(ix):
   v=(0.,.45,.8,1.)[i%4]; cols=':1' if j==0 else '1:'; name='ys' if j==0 else 'ts'; f.write(f'    base[:,{cols}] = {1-v!r} * base[:,{cols}] + {v!r} * {name}[{i//4}]\n')
  f.write("    y=base[:,0]; state=base[:,1:]; last=state[:,-1]\n    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),last.mean(),last.std()])}\n")
