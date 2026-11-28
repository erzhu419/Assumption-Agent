import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
np.set_printoptions(precision=4,suppress=True,linewidth=150)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('times',s['eval_times'])
for k in ('panel_a','panel_b','panel_c'):
 p=s[k];u=p['u']; f=np.c_[np.ones(len(u)),u]; print(k)
 for key,v in p.items():
  if key=='u':continue
  for trans in (lambda v:v,np.tanh,lambda v:np.arctanh(np.clip(v,-.999,.999)),np.arcsinh):
   w=trans(v); b=np.linalg.lstsq(f,w,rcond=None)[0]
   for i in range(5):
    r=np.linalg.norm(w-f@b,axis=1); ii=np.argsort(r)[:int(len(r)*.85)];b=np.linalg.lstsq(f[ii],w[ii],rcond=None)[0]
   print('res',np.quantile(np.linalg.norm(w-f@b,axis=1),[0,.5,.8,.9,1]), 'coef',b.T if trans(v) is v else '')
x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']];print('sv',np.linalg.svd(Y-Y.mean(0),compute_uv=False));
for d in range(1,6):
 X=PolynomialFeatures(d).fit_transform(x)
 for a in (.0001,.01,.1,1):
  m=Ridge(a).fit(X[:52],Y[:52]);e=Y[52:]-m.predict(X[52:]);print(d,a,np.mean(abs(e),axis=0))
for d in (1,2,3):
 f=PolynomialFeatures(d).fit_transform(np.c_[x,Y[:,[1,4,8]]]); b=Ridge(.01).fit(f[:52],Y[:52,0]); print('oracle y',d,np.mean(abs(Y[52:,0]-b.predict(f[52:]))))
