import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=130)
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
poly=lambda x,d:PolynomialFeatures(d).fit_transform(x)
print('times',s['eval_times'])
for k in ('panel_a','panel_b','panel_c'):
 p=s[k];u=p['u'];print(k)
 for v in [v for a,v in p.items() if a!='u']:
  for d in [1,2]:
   D=poly(u,d);inds=np.arange(len(u));
   for _ in range(20):
    C=fit(D[inds],v[inds]);r=np.linalg.norm(D@C-v,axis=1);inds=np.argsort(r)[:int(.8*len(u))]
   print(d,'quantiles',np.quantile(r,[0,.25,.5,.75,.9,1]),'coef',C if d==1 else '')
tr=s['train']; x=tr['x'];T=tr['traj']; y=tr['y']; Z=np.c_[y,T];
for d in [1,2,3]:
 D=poly(x,d); C=fit(D[:52],Z[:52]);r=D@C-Z;print('poly',d,'train/test mae',np.mean(abs(r[:52]),0),np.mean(abs(r[52:]),0));
 if d==2:print('quadratic coeff y, trajectory first/last',C[:,[0,1,-1]])
U,S,V=np.linalg.svd(T,full_matrices=False);print('temporal basis',S,V[:2]);print('temporal recurrence',fit(T[:,:-2].ravel()[:,None]**0,T[:,2:].ravel()) if False else fit(np.c_[T[:,1:-1].ravel(),T[:,:-2].ravel(),np.repeat(T[:,-1],6)],T[:,2:].ravel()))
