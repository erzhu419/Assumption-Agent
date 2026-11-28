import pickle,numpy as np
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a,b,c=[s['panel_'+j] for j in 'abc']
def ff(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def rm(x,y):return np.mean((y-x@fit(x,y))**2)**.5
u=a['u']; v=a['v']; k=np.linalg.norm(v-u,axis=1)<4; u,v=u[k],v[k]; print('A clean',len(u), 'range',u.min(0),u.max(0)); print('A fit',fit(ff(u),v));
for d in (2,3):
 for fs in [np.c_[ff(u),u**d], np.c_[ff(u),*[np.prod(u[:,j],axis=1) for j in cr(range(4),d)]],np.c_[ff(u),u*u,u**d]]:
  print('A poly',fs.shape[1],rm(fs,v));
  if fs.shape[1] in (9,13):print(fit(fs,v).round(3))
for j in ('v1','v2'):
 u=c['u'];v=c[j];
 if j=='v1':
  for f in [lambda x:np.arctan(2.15*x),lambda x:np.arcsinh(2.15*x),lambda x:np.tanh(2.15*x),lambda x:x/(1+.22*x*x)**.5]:
   g=np.c_[ff(u),f(u)];print('C odd',rm(g,v),fit(g,v).round(3))
 else:
  gg=lambda x,k:x+1.5*np.log1p(k*x*x); o=minimize_scalar(lambda k:rm(ff(gg(u,k)),v),bounds=(.1,.7),method='bounded');print('C log',o.x,o.fun)
t=s['eval_times']; z=s['train']['traj']; print('times',t); funs={'exp':lambda t,k:1-np.exp(-k*t),'rat':lambda t,k:t/(1+k*t),'log':lambda t,k:np.log1p(k*t)}
for kn,f in funs.items():
 for ln,g in funs.items():
  def basis(k):return np.array([f(t,k[0]),g(t,k[1])]).T
  o=least_squares(lambda k:(z.T-basis(k)@fit(basis(k),z.T)).ravel(),[1,5],bounds=(.01,30)); print(kn,ln,o.x,np.mean(o.fun**2)**.5)
