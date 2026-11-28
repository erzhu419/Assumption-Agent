import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def af(a): return np.column_stack([np.ones(len(a)),a])
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def rm(a):return np.mean(a*a)**.5
np.set_printoptions(precision=5,suppress=True)
a=s['panel_a'];u,v=a['u'],a['v'];
for j,(x,y) in enumerate(zip(u.T,v.T)):
 p=least_squares(lambda p:af(x[:,None])@p-y,[0,1],loss='cauchy',f_scale=.06).x;r=y-af(x[:,None])@p; print('a column',j,p,'res q',np.quantile(abs(r),[0,.25,.5,.75,.85,.95,1])); ix=np.argsort(x);print(np.c_[x[ix],r[ix]][::3].T)
for key,ki,ko in [('panel_b','u','v'),('panel_c','u','v1'),('panel_c','u','v2'),('panel_c','v1','v2')]:
 u,v=s[key][ki],s[key][ko];
 print('\n',key,ki,ko,'affine',rm(af(u)@fit(af(u),v)-v)); print(fit(af(u),v))
 def powr(x,p):a,b,c=p;return np.sign(x)*abs(x)**b/(1+a*abs(x)**c)
 def soft(x,p):a,b=p;return np.sign(x)*np.maximum(abs(x)-a,0)**b
 def cube(x,p):a,b=p;return (x**3+b*x)/np.sqrt(1+a*abs(x)**3)
 def tanh(x,p):return np.tanh(x*p[0])
 def odd(x,p):a,b=p;return (x+b*x**3)/(1+a*x*x)
 for fn,p,lo,hi in [(powr,[.22,3,4],[0,.15,.1],[5,8,16]),(soft,[.22,1.25],[0,.1],[.7,8]),(cube,[.22,0],[0,-2],[10,3]),(tanh,[1],[.001],[8]),(odd,[.22,0],[0,-2],[10,3])]:
  for direction in [0,1]:
   def rr(p):
    U,V=(af(fn(u,p)),v) if not direction else (af(u),fn(v,p));return (U@fit(U,V)-V)/max(rm(V-V.mean(0)),1e-15)
   pp=least_squares(lambda p:rr(p).ravel(),p,bounds=(lo,hi),max_nfev=150).x
   print(fn.__name__,direction,pp,rm(rr(pp)))
print('traj sv',np.linalg.svd(s['train']['traj'],compute_uv=False))
