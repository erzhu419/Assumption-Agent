import numpy as np,pickle,itertools
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=4,suppress=True,linewidth=170)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(u):return np.c_[np.ones(len(u)),u]
def robust(a,v):
 c=ls(a,v)
 for i in range(40):
  r=np.linalg.norm(a@c-v,axis=1); w=np.sqrt(np.minimum(1,1.2*np.median(r)/np.maximum(r,1e-12)));c=ls(w[:,None]*a,w[:,None]*v)
 g=np.linalg.norm(a@c-v,axis=1)<4*np.median(np.linalg.norm(a@c-v,axis=1))+1e-10
 return ls(a[g],v[g]),g
for nm in ('panel_a','panel_b','panel_c'):
 p=s[nm];u=p['u']; poly=PolynomialFeatures(2);f=poly.fit_transform(u)
 funcs={'aff':aff(u),'quad':f,'sq':np.c_[aff(u),u*u],'pair':np.c_[aff(u),f[:,[6,7,8,10,11,13]]],'tanh':aff(np.tanh(u)),'sin':aff(np.sin(u)),'abs':np.c_[aff(u),abs(u)],'cube':np.c_[aff(u),u**3]}
 for key,v in p.items():
  if key=='u':continue
  print(nm,key)
  for name,a in funcs.items():
   c,g=robust(a,v); ag=a[g]; vg=v[g];res=np.linalg.norm(ag@c-vg,axis=1);h=np.diag(ag@np.linalg.pinv(ag));print(name,g.sum(),np.median(res),np.median(res/(1-h)))
   if name=='quad':print(list(poly.get_feature_names_out()),c.T)
   if name=='pair':print(c.T)
x=s['train']['x']; y=s['train']['y'];t=s['train']['traj']; poly=PolynomialFeatures(2);f=poly.fit_transform(x);q=ls(f[:52],np.c_[y,t][ :52]); print('train quadratic',list(poly.get_feature_names_out()),q[:,[0,1,8]].T); print('ranges',x.min(0),x.max(0));print('rows x,y,terminal,residuals');print(np.c_[x,y,t[:,-1],(np.c_[y,t]-f@q)[:,[0,-1]]])
