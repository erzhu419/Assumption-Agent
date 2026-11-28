import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def af(x): return np.c_[np.ones(len(x)),x]
def clip(a,v):
 m=np.ones(len(a),bool)
 for _ in range(12):
  w=ls(a[m],v[m]); e=np.linalg.norm(a@w-v,axis=1); n=e<=max(1e-8,3*np.median(e[m]));
  if n.sum()<=a.shape[1]+2 or np.array_equal(n,m): break
  m=n
 return w,e,m
print('telemetry',s['telemetry_vec'])
for p in ('panel_a','panel_b','panel_c'):
 u=s[p]['u']; pair=np.array([u[:,i]*u[:,j] for i,j in itertools.combinations(range(4),2)]).T
 for key,v in s[p].items():
  if key=='u':continue
  print(p,key)
  h={'aff':u[:,:0],'sq':u**2,'cube':u**3,'sqcube':np.c_[u**2,u**3],'quad':np.c_[u**2,pair], 'absq':u*np.abs(u), 'sat1':u/(1+1.25*abs(u)), 'sat2':u**2/(1+.22*u**2),'tanh':np.tanh(1.25*u),'sin':np.sin(1.25*u)}
  for name,z in h.items():
   w,e,m=clip(np.c_[af(u),z],v); print(name,m.sum(), 'med %.5f mean %.5f max %.5f'%(np.median(e[m]),np.mean(e[m]),max(e[m])), 'excluded',np.where(~m)[0]);
   if name in ['aff','sqcube'] and p=='panel_a':print(w)
  for name,fn in [('sat1',lambda x,k:x/(1+k*abs(x))),('sat2',lambda x,k:x**2/(1+k*x*x)),('tanh',lambda x,k:np.tanh(k*x)),('sin',lambda x,k:np.sin(k*x))]:
   def fit(k,all=False):
    a=np.c_[af(u),fn(u,k[0])];w,e,m=clip(a,v);
    return (w,e,m) if all else np.where(m[:,None],a@w-v,0).ravel()
   q=least_squares(fit,[.8],bounds=(.001,5));w,e,m=fit(q.x,True);print('opt',name,q.x,m.sum(),np.median(e[m]),max(e[m]));
   if np.median(e[m])<.02:print(w)
