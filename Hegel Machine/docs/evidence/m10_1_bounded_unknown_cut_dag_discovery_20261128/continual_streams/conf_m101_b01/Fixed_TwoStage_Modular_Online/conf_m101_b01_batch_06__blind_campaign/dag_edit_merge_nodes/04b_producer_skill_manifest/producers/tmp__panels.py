import numpy as np,pickle
from itertools import combinations
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
i,j=np.triu_indices(4,1); triples=list(combinations(range(4),3))
def features(u):
 return dict(lin=np.c_[np.ones(len(u)),u],cross=u[:,i]*u[:,j],sq=u*u,cube=u**3,triple=np.prod(u[:,triples],axis=2))
def robust(a,v):
 w=np.ones(len(a))
 for _ in range(30):
  c=np.linalg.lstsq(a*w[:,None],v*w[:,None],rcond=None)[0];e=np.linalg.norm(v-a@c,axis=1); w=1/np.sqrt(1+(e/max(1e-4,1.8*np.median(e)))**2)
 return c,e
if __name__=='__main__':
 u,v=s['panel_a']['u'],s['panel_a']['v']; f=features(u)
 for keys in [('lin',),('lin','cross'),('lin','sq'),('lin','cross','sq'),('lin','cross','sq','cube'),('lin','cross','sq','triple')]:
  a=np.column_stack([f[k] for k in keys]); c,e=robust(a,v); inds=np.argsort(e)[:-3]; c=np.linalg.lstsq(a[inds],v[inds],rcond=None)[0]; print('A',keys,np.quantile(np.linalg.norm(v-a@c,axis=1),[.5,.9,.94,1]),'out',np.argsort(e)[-3:]);
  if len(keys)<=3:print(c.round(3))
 for p,k in [('panel_b','v'),('panel_c','v2')]:
  u,v=s[p]['u'],s[p][k]; f=features(u); print(p,k)
  for name,fun in [('tanh',lambda u,a:np.tanh(a*u)),('sin',lambda u,a:np.sin(a*u)),('soft',lambda u,a:u/(1+a*u*u)),('sqrt',lambda u,a:u/np.sqrt(1+a*u*u)),('exp',lambda u,a:np.exp(-a*u*u)),('cos',lambda u,a:np.cos(a*u)),('log',lambda u,a:np.log1p(a*u*u)),('atan',lambda u,a:np.arctan(a*u))]:
   def score(a):
    F=np.c_[f['lin'],fun(u,a)]; H=np.linalg.pinv(F); return np.mean(((v-F@(H@v))/(1-np.sum(F*H.T,axis=1))[:,None])**2)
   opt=minimize_scalar(score,bounds=(.01,3),method='bounded'); print(name,round(opt.x,5),round(np.sqrt(opt.fun),6))
