import pickle
from itertools import combinations
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=120)
for pn,vn in [('panel_a','v'),('panel_c','v2')]:
 p=s[pn]; u=p['u']; v=p[vn]; one=np.ones(len(u)); L=np.column_stack([one,u]); ix=np.arange(len(u));
 if pn=='panel_a':
  w=np.ones(len(u))
  for _ in range(20):
   c=np.linalg.lstsq(L*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(v-L@c,axis=1); w=1/np.hypot(1,r/(2*np.median(r)))
  ix=np.sort(np.argsort(r)[:-3]); print('A retained',len(ix),'residual gap',np.sort(r)[-4:])
 prod=np.column_stack([u[:,i]*u[:,j] for i,j in combinations(range(4),2)])
 for lab,D in [('linear',L),('diagonal2',np.c_[L,u**2]),('quad',np.c_[L,u**2,prod]),('sep3',np.c_[L,u**2,u**3]),('quad+cube',np.c_[L,u**2,prod,u**3]),('sep4',np.c_[L,u**2,u**3,u**4])]:
  d=D[ix]; y=v[ix]; c=np.linalg.lstsq(d,y,rcond=None)[0]; e=y-d@c; q=np.linalg.qr(d,mode='reduced')[0]; loo=e/(1-(q*q).sum(1)[:,None]); print(pn,lab,'rmse',np.sqrt(np.mean(e**2)),'loo',np.sqrt(np.mean(loo**2)),'row coeff norms',np.round(np.linalg.norm(c,axis=1),5));
  if lab in ('quad','sep3'): print(c)
 for fun in [np.sin,np.tanh,np.exp]:
  for freq in [.5,1,1.5,2,3]:
   d=np.c_[L,fun(freq*u),np.cos(freq*u)][ix]; y=v[ix]; c=np.linalg.lstsq(d,y,rcond=None)[0]; e=y-d@c; q=np.linalg.qr(d,mode='reduced')[0]; print(pn,fun.__name__,freq,'loo',np.sqrt(np.mean((e/(1-(q*q).sum(1)[:,None]))**2)))
