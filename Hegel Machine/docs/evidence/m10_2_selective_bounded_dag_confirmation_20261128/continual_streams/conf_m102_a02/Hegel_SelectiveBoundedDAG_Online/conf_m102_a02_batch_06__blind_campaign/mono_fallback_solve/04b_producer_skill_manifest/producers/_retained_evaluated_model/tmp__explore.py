import numpy as np,pickle,itertools,importlib.util
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return lstsq(a,b)[0]
def poly(x,d):return PolynomialFeatures(d).fit_transform(x)
np.set_printoptions(precision=4,suppress=True)
for name in ('a','b','c'):
 p=s['panel_'+name]; u=p['u'];
 for key in list(p)[1:]:
  v=p[key];keep=np.linalg.norm(v,axis=1)<4 if name=='a' else np.ones(len(u),bool)
  for d in [1,2,3]:
   f=poly(u,d); g=f[keep]; b=ls(g,v[keep]); res=g@b-v[keep]; loo=res/(1-np.einsum('ij,ji->i',g,np.linalg.pinv(g)))[:,None]; print(name,key,d,'rmse',np.mean(res**2)**.5,'LOO',np.mean(loo**2)**.5); 
   if d==2:print(b.T)
t=s['train']; z=t['traj']; y=t['y']; x=t['x'];
for i in range(4):
 spec=importlib.util.spec_from_file_location('m',f'/tmp/hegel_continual_archive/conf_m102_a02_batch_0{i}_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m); o=m.solve_batch(dict(s,train={k:v[:52] if k!='summary' else v for k,v in t.items()},test={'x':x[52:]}));print('arch',i,np.mean(abs(o['y_hat']-y[52:])),np.mean(abs(o['state_pred']-z[52:])))
