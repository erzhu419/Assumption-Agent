import numpy as np,pickle
from scipy.linalg import lstsq
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=180)
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; Y=np.c_[y,t]; print('time',s['eval_times']);print('trajectory SVD',np.linalg.svd(t,compute_uv=False));print('outputs std',Y.std(0));
ops=[]
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; U=np.c_[np.ones(len(u)),u];
 for j in p:
  if j=='u':continue
  v=p[j]; mask=np.ones(len(u),bool)
  for _ in range(5 if k=='panel_a' else 1):
   w=lstsq(U[mask],v[mask])[0];err=np.linalg.norm(U@w-v,axis=1);mask=err<=np.partition(err,len(u)-8)[len(u)-8] if k=='panel_a' else mask
  print(k,j,'coeff',w,'residual quantiles',np.quantile(err,[0,.25,.5,.75,.85,.95,1])); print('quadratic residual',np.std(PolynomialFeatures(2).fit_transform(u)@lstsq(PolynomialFeatures(2).fit_transform(u),v)[0]-v,0));ops.append(w)
for label,z in [('raw',x)]+[(str(i),x@w[1:]+w[0]) for i,w in enumerate(ops)]:
 for d in range(1,5):
  a=PolynomialFeatures(d).fit_transform(z); model=Ridge(alpha=.015).fit(a[:52],Y[:52]);e=abs(model.predict(a[52:])-Y[52:]); print(label,d,e.mean(0))
for idx in [[0,1,3,5,7],list(range(8))]:
 a=np.c_[np.ones(len(t)),t[:,idx]]; w=lstsq(a[:52],y[:52])[0]; print('readout traj',idx,w,np.mean(abs(y[52:]-a[52:]@w)))
