import numpy as np,pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; print('times',t)
f=lambda x,d:PolynomialFeatures(d).fit_transform(x)
np.set_printoptions(precision=4,suppress=True,linewidth=155)
for k in ('panel_a','panel_b','panel_c'):
 u=s[k]['u']
 for key,v in s[k].items():
  if key=='u':continue
  print(k,key)
  for d in (1,2,3):
   a=f(u,d); b=np.linalg.lstsq(a,v,rcond=None)[0]; r=v-a@b
   if d==1:
    keep=np.ones(len(u),bool)
    for _ in range(5):
     b=np.linalg.lstsq(a[keep],v[keep],rcond=None)[0]; r=v-a@b; keep=np.linalg.norm(r,axis=1)<max(.02,2.5*np.median(np.linalg.norm(r,axis=1)))
    print('affine coefficients',b,'sv',np.linalg.svd(b[1:],compute_uv=False),'norms',np.sort(np.linalg.norm(r,axis=1)))
   print(d,np.sqrt((r*r).mean(axis=0)))
X=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; print('joint singular values',np.linalg.svd(Y-Y.mean(0),compute_uv=False)); print('correlations',np.corrcoef(np.c_[X,Y].T)[:5]); print('traj coeffs',np.linalg.lstsq(f(X,1),Y,rcond=None)[0]);
for d in (1,2,3,4,5):
 a=f(X,d)
 for l in (.00001,.001,.01,.1,1):
  m=Ridge(l).fit(a[:52],Y[:52]); e=abs(m.predict(a[52:])-Y[52:]); print('poly',d,l,e.mean(0))
print('train ranges',X.min(0), X.max(0),'test ranges', s['test']['x'].min(0),s['test']['x'].max(0))
