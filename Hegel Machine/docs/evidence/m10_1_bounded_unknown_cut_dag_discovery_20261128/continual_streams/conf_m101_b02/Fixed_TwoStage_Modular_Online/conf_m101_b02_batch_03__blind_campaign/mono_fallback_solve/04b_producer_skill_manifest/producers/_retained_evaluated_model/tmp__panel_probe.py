import numpy as np,pickle
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=4,suppress=True)
print('times',s['eval_times'],'telemetry',s['telemetry_vec'])
def fit(F,v,robust):
 w=np.linalg.lstsq(F,v,rcond=None)[0]
 for j in range(35 if robust else 0):
  e=np.linalg.norm(v-F@w,axis=1); q=1/np.maximum(e,.00025); w=np.linalg.lstsq(F*q[:,None]**.5,v*q[:,None]**.5,rcond=None)[0]
 return w
for p in ['panel_a','panel_c']:
 a=s[p];u=a['u'];n=len(u)
 for k,v in a.items():
  if k=='u':continue
  for name,F in [('lin',np.c_[np.ones(n),u]),('diag23',np.c_[np.ones(n),u,u**2,u**3]),('quad',PolynomialFeatures(2).fit_transform(u)),('quad+cubes',np.c_[PolynomialFeatures(2).fit_transform(u),u**3]),('cubic',PolynomialFeatures(3).fit_transform(u))]:
   if F.shape[1]>=n-2:continue
   w=fit(F,v,p=='panel_a');e=np.linalg.norm(v-F@w,axis=1)
   r=[]
   for i in range(n):
    m=np.arange(n)!=i;r.append(np.linalg.norm(F[i]@fit(F[m],v[m],p=='panel_a')-v[i]))
   print(p,k,name,'fit',np.quantile(e,[.25,.5,.75,.9,1]),'loo',np.quantile(r,[.25,.5,.75,.9,1]));
   if name=='diag23' and p=='panel_a':print(w)
x=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];_,sv,vt=np.linalg.svd(T-T.mean(0),full_matrices=False)
print('x percentiles',np.quantile(x,[0,.1,.5,.9,1],axis=0));print('time mean and first components',T.mean(0),vt[:2]);print('y std',y.std())
for deg in [1,2,3]:
 F=PolynomialFeatures(deg).fit_transform(x); z=np.c_[y,T]; W=np.linalg.lstsq(F[:52],z[:52],rcond=None)[0];e=np.abs(z[52:]-F[52:]@W);print('raw poly',deg,e[:,0].mean(),e[:,1:].mean())
