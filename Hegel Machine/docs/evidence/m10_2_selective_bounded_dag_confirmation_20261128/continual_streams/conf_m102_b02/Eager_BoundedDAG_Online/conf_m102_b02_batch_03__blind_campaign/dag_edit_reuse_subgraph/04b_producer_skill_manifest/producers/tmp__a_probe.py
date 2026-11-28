import numpy as np,pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; D=lambda a:np.c_[np.ones(len(a)),a]
a=D(u); c=np.linalg.lstsq(a,v,rcond=None)[0]; r=np.linalg.norm(a@c-v,axis=1); good=r<3; print('outliers',np.where(~good)[0],v[~good]); u=u[good];v=v[good];
print('u,v ranges',u.min(0),u.max(0),v.min(0),v.max(0));
fs={'aff':lambda x:D(x),'sep2':lambda x:D(np.c_[x,x*x]),'sep3':lambda x:D(np.c_[x,x*x,x**3]),'poly2':lambda x:D(np.c_[x,*[np.prod(x[:,i],axis=1) for i in combinations_with_replacement(range(4),2)]]),'poly3':lambda x:D(np.c_[x,*[np.prod(x[:,i],axis=1) for d in (2,3) for i in combinations_with_replacement(range(4),d)]]),'sinh':lambda x:D(np.c_[x,np.sinh(x)]),'asinh':lambda x:D(np.c_[x,np.arcsinh(x)]),'tanh':lambda x:D(np.c_[x,np.tanh(x)]),'sin':lambda x:D(np.c_[x,np.sin(x)])}
for name,x,y in [('forward',u,v),('inverse',v,u)]:
 for k,f in fs.items():
  a=f(x); c=np.linalg.lstsq(a,y,rcond=None)[0]; r=y-a@c; h=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,axis=1); loo=r/(1-h[:,None]); print(name,k,np.round(np.sqrt(np.mean(r*r,axis=0)),6),np.round(np.sqrt(np.mean(loo*loo,axis=0)),6));
  if k in ('aff','sep3') and name=='forward': print(np.round(c,4)); print(np.round(np.c_[x,r][:9],4))
