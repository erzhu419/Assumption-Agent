import importlib.util,pickle,numpy as np
from itertools import combinations
p='/holdout_task/node_01.py'; spec=importlib.util.spec_from_file_location('candidate',p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=np.c_[t['y'],t['traj']]; a,m=mod._features(s,x); z,g,v=a[:,1:5],a[:,5:9],a[:,9:13]; n=len(x)
folds=[(np.arange(n-20),np.arange(n-20,n))]; rng=np.random.default_rng(8451)
for _ in range(60):
 p=rng.permutation(n); folds.append((p[20:],p[:20]))
for score in (np.linalg.norm(x,axis=1),np.max(abs(z),axis=1),np.linalg.norm(v,axis=1)):
 p=np.argsort(score); folds.append((p[:-12],p[-12:]))
def profile(des,alpha,folds):
 out=[]
 for tr,te in folds:
  core=des[:,:a.shape[1]]; b=des[:,a.shape[1]:]; c=mod._fit(core[tr],y[tr,0]); py=core[te]@c
  if b.shape[1]:
   r=b-core@mod._fit(core[tr],b[tr]); scale=np.maximum(np.sqrt(np.sum(r[tr]**2,axis=0)),1e-10); r=r/scale
   py+=r[te]@np.linalg.solve(r[tr].T@r[tr]+alpha*np.eye(r.shape[1]),r[tr].T@(y[tr,0]-core[tr]@c))
  pt=a[te,:m]@mod._fit(a[tr,:m],y[tr,1:]); err=abs(np.c_[py,pt]-y[te]); out.append(np.r_[err[:,0].mean(),err[:,1:].mean(),err[:,1:].mean(0)])
 return np.array(out)
def report(out):
 print('public',out[0,:2].round(6),'random_mean',out[1:-3,:2].mean(0).round(6),'random_worst',out[1:-3,:2].max(0).round(6),'boundary',out[-3:,:2].round(6).tolist(),'worst_horizon',round(out[:,2:].max(),6))
report(profile(a,0,folds))
groups={'adjacent_g':g*np.roll(g,1,axis=1),'latent_sq':z*z,'latent_v':z*v,'v_sq':v*v,'g_sq':g*g,'roll_latent_v':z*np.roll(v,1,axis=1),'latent_softplus':np.logaddexp(0,2.2*z),'v_saturated':v/(1+v*v),'latent_rat':z/(1+1.25*z*z)}
res=[]
for k,b in groups.items():
 for alpha in (0.03,.3,1.):
  out=profile(np.c_[a,b],alpha,folds); res.append((out[1:-3,0].mean(),k,alpha,out))
for mean,k,alpha,out in sorted(res,key=lambda r:r[0])[:6]:
 print(k,alpha); report(out)
