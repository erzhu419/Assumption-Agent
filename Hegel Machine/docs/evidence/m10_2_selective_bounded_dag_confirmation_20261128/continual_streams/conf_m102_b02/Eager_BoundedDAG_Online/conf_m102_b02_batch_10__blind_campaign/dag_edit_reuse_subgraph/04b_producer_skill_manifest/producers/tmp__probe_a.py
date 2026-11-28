import pickle,numpy as np
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; f=s['telemetry_vec'][1]
rng=np.random.default_rng(31)
print('paired sample:',np.c_[u,v][:10].round(3)); print('ranges:',u.min(),u.max(),v.min(),v.max())
def cols(x):
 return {'lin':x,'sin':np.sin(f*x),'cos':np.cos(f*x)-1,'adj':x*np.roll(x,-1,axis=1),'cross':np.stack([x[:,i]*x[:,j] for i,j in combinations(range(4),2)],1),'cube':x**3,'sq':x*x,'sinadj':np.sin(f*x*np.roll(x,-1,axis=1))}
for spec in ['lin','lin adj cube','lin sin','lin sin adj','lin sin cos','lin sin cos adj','lin sin cross sq','lin adj sinadj','lin sin adj cube']:
 c=cols(u); F=np.c_[np.ones(len(u)),*[c[k] for k in spec.split()]]; n=len(F); best=None
 for it in range(400):
  keep=np.arange(n) if it==0 else rng.choice(n,max(F.shape[1]+1,20),False)
  for j in range(4):
   w=np.linalg.lstsq(F[keep],v[keep],rcond=None)[0]; e=np.linalg.norm(F@w-v,axis=1); keep=np.argsort(e)[:36]
  score=np.median(e)
  if best is None or score<best[0]:best=(score,np.quantile(e,[.1,.25,.5,.65,.75,.8,.85,.9,1]),w,e)
 print(spec,best[1].round(5), 'coef norms',np.linalg.norm(best[2],axis=1).round(4))
 if spec=='lin sin adj':print(best[2].round(4))
