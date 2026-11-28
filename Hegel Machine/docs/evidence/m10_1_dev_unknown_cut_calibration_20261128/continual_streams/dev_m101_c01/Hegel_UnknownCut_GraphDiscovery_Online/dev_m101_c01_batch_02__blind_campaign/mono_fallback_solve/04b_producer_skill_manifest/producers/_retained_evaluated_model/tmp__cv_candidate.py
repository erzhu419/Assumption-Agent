import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _predict
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; ids=np.arange(len(x)); order=np.argsort(np.linalg.norm(x,axis=1)); rng=np.random.default_rng(511)
splits=[('56/16',ids[:56],ids[56:])]+[(str(i),p[:56],p[56:]) for i,p in enumerate([rng.permutation(ids) for _ in range(5)])]+[('outer',order[:56],order[56:]),('test-range outer',np.r_[order[:49],order[65:]],order[49:65])]
for a,b,c in [('classic','classic','joint'),('extended','extended','joint'),('classic','classic','amp'),('extended','extended','amp')]:
 out=[]
 for name,i,j in splits:
  f=dict(s,train={k:v[i] for k,v in t.items() if k!='summary'},test={'x':x[j]}); y,z=_predict(f,a,b,c)
  out.append([np.mean(abs(y-t['y'][j])),np.mean(abs(z-t['traj'][j])),np.mean(abs(z[:,-1]-t['traj'][j,-1]))])
 print(a,b,c,np.round(np.array(out).T,5),flush=True)
