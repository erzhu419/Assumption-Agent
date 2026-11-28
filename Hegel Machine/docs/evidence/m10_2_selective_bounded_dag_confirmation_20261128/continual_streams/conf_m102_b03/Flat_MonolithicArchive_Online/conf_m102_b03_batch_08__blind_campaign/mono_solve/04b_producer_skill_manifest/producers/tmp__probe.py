import numpy as np,pickle,itertools as it
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-11)[0]
def poly(x,k):
 return np.column_stack([np.prod(x[:,c],axis=1) for d in range(k+1) for c in it.combinations_with_replacement(range(4),d)])
def D(x): return np.c_[np.ones(len(x)),x]
def adj(x):return x*np.roll(x,-1,axis=1)
A=s['panel_a']; u,v=A['u'],A['v']; good=np.max(abs(v-u),1)<1
print('A good',sum(good));
for j in range(4):
 a=u[:,j]; b=np.roll(u,-1,axis=1)[:,j]; c=np.roll(u,1,axis=1)[:,j];
 for name,d in [('aff',D(u)),('p2',poly(u,2)),('prev',D(np.c_[a,c,a*c,a**3])),('next',D(np.c_[a,b,a*b,a**3])),('self',D(np.c_[a,a*a,a**3])),('p3',poly(u,3))]:
  h=ls(d[good],v[good,j]); print(j,name,np.std((d@h-v[:,j])[good]),np.round(h,5) if len(h)<=5 else '')
def cand(x,k):
 return dict(lin=D(x),quad=D(adj(x)),prev=D(x*np.roll(x,1,axis=1)),square=D(x*x),cube=D(x**3),odd=D(np.tanh(k*x)),sin=D(np.sin(k*x)),cos=D(np.cos(k*x)),soft=D(np.logaddexp(0,k*x)),rat=D(x/(1+k*x*x)),sat=D(x*x/(1+k*x*x)),linquad=D(np.c_[x,adj(x)]),p2=poly(x,2),linpol=D(np.c_[x,x*x,x**3]),exp=D(np.exp(k*x)),sig=D(1/(1+np.exp(-k*x))))
for pn in ['panel_b','panel_c']:
 p=s[pn];u=p['u'];
 for n,v in p.items():
  if n=='u':continue
  print(pn,n)
  def err(d):
   h=ls(d,v); er=d@h-v; lev=np.sum(d*np.linalg.pinv(d).T,axis=1);return np.sqrt(np.mean((er/np.maximum(1-lev,1e-9)[:,None])**2))
  for key,a in cand(u,2.4).items():
   if key in ('odd','sin','cos','soft','rat','sat','exp','sig'):
    f=lambda k:err(cand(u,k)[key]); r=minimize_scalar(f,bounds=(.05,4),method='bounded'); print(key,round(r.fun,7),round(r.x,5))
   elif key!='prev': print(key,round(err(a),7))
