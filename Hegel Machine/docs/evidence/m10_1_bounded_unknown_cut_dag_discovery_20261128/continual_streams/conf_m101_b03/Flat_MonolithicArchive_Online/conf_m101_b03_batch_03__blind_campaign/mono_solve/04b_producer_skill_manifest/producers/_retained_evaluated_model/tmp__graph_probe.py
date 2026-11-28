import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; T=s['train']['traj'][:52].T
np.set_printoptions(precision=5,suppress=True)
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,k):return np.column_stack([np.prod(x[:,ij],axis=1) for ij in combinations_with_replacement(range(x.shape[1]),k)])
def cal(x):return np.c_[aff(x),poly(x,2),x**3]
a=s['panel_a']; u=a['u']; v=a['v']; keep=np.max(abs(v),axis=1)<3
p=fit(cal(u)[keep],v[keep]); print('cal terms',p); print('clean residual',np.sort(np.linalg.norm(cal(u)@p-v,axis=1)))
b=s['panel_b'];print('B affine',fit(aff(b['u']),b['v']))
c=s['panel_c']; u=c['u'];
for name in ['v1','v2']:
 v=c[name];
 for q,D in [('quad',np.c_[aff(u),poly(u,2)]),('only quad',aff(poly(u,2))),('diagonal',np.c_[aff(u),u**2]),('cycles',np.c_[aff(u),u[:,[0,0,1,2]]*u[:,[1,3,2,3]]])]: print(name,q,np.linalg.norm(D@fit(D,v)-v));
 for trig in [np.cos,np.sin]:
  for add in [False,True]:
   def design(f):return aff(np.c_[u,trig(f*u)] if add else trig(f*u))
   def loss(f):
    D=design(f);return np.linalg.norm(D@fit(D,v)-v)
   g=np.linspace(.1,10,300); i=np.argmin([loss(f) for f in g]); r=minimize_scalar(loss,bounds=(g[max(i-1,0)],g[min(i+1,len(g)-1)]),method='bounded');print(name,trig.__name__,add,r.x,r.fun)
for typ in ['exp2','exp3','exp2+linear','exp2+const','cascade']:
 def basis(q):
  E=-np.expm1(-t[:,None]*q); 
  if typ=='exp2+linear':return np.c_[E,t]
  if typ=='exp2+const':return np.c_[E,np.ones_like(t)]
  if typ=='cascade':return np.c_[E,t*np.exp(-q[1]*t)]
  return E
 def err(q):
  E=basis(q);return (E@fit(E,T)-T).ravel()
 q=least_squares(err,[.2,.9,2.5] if typ=='exp3' else [.3,1.8],bounds=(.0001,12),max_nfev=300).x;print(typ,q,np.sqrt(np.mean(err(q)**2)))
