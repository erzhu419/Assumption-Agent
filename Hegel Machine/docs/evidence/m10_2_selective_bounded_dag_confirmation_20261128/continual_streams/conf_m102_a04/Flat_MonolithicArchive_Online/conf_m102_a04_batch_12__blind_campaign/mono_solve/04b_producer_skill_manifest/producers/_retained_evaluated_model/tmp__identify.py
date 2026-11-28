import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
lin=lambda f,v:np.linalg.lstsq(f,v,rcond=1e-10)[0]
def one(x):return np.c_[np.ones(len(x)),x]
p=s['panel_a']; mask=np.linalg.norm(p['v'],axis=1)<4; u,v=p['u'][mask],p['v'][mask]; pp=PolynomialFeatures(3); f=pp.fit_transform(u); w=lin(f,v)
for j in range(4):print('A cubic',j,[(k,round(c,4)) for k,c in sorted(zip(pp.get_feature_names_out(),w[:,j]),key=lambda z:-abs(z[1]))[:14]])
for key in ['v1','v2']:
 p=s['panel_c'];u0,v0=p['u'],p[key]; pp=PolynomialFeatures(3); f=pp.fit_transform(u0); names=pp.get_feature_names_out()
 for d in [2,3]:
  ids=np.flatnonzero(pp.powers_.sum(1)<=d); print(key,d,np.sqrt(np.mean((f[:,ids]@lin(f[:,ids],v0)-v0)**2)))
  if d==2:print(np.c_[names[ids],np.round(lin(f[:,ids],v0),4)])
 for j in range(4):
  ids=[]; r=v0[:,j].copy(); norms=np.linalg.norm(f,axis=0)
  for k in range(18):
   scores=abs(f.T@r)/norms;scores[ids]=-1; ids.append(np.argmax(scores)); b=lin(f[:,ids],v0[:,j]);r=v0[:,j]-f[:,ids]@b
   if np.std(r)<.0003:break
  print(key,j,np.std(r),[(names[i],round(c,4)) for i,c in zip(ids,b)])
