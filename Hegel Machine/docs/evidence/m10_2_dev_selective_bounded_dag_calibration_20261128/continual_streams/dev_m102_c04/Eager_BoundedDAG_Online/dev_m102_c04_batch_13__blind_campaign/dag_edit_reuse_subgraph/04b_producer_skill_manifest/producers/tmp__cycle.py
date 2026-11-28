import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_c']['u']; v=s['panel_c']['v2']; a=np.c_[np.ones(len(x)),x]; out=[]
fun={'sin':lambda t,k:np.sin(k*t),'tanh':lambda t,k:np.tanh(k*t),'soft':lambda t,k:t/(1+k*abs(t)),'rat':lambda t,k:t/(1+k*t*t),'sqrat':lambda t,k:t*t/(1+k*t*t),'cos':lambda t,k:np.cos(k*t),'atan':lambda t,k:np.arctan(k*t),'asinh':lambda t,k:np.arcsinh(k*t),'clip':lambda t,k:np.clip(t,-1/k,1/k),'swish':lambda t,k:t*np.tanh(k*t)}
patterns=[(j,p) for j in range(1,4) for p in itertools.permutations(range(4)) if p[0]==0]; grid=[.001,.08,.22,.5,1,1.6,2.15,3,4,6]
def fit(name,fn):
 def rm(k):
  f=fn(k); w=np.linalg.lstsq(f,v,rcond=1e-9)[0]; return np.mean((v-f@w)**2)
 z=[rm(k) for k in grid]; j=np.argmin(z); o=minimize_scalar(rm,bounds=(grid[max(j-1,0)],grid[min(j+1,len(grid)-1)]),method='bounded'); out.append((o.fun**.5,o.x,name))
for step,p in patterns:
 xx=x[:,p]; y=np.roll(xx,-step,axis=1); z=np.roll(xx,-2*step,axis=1); t=np.roll(xx,-3*step,axis=1)
 bases={'prod':xx*y,'sum':xx+y,'diff':xx-y,'sqdiff':xx**2-y**2,'triple':xx*y*z,'sumprod':xx*(y+z),'diffprod':xx*(y-z),'prodsum':xx*y+z*t,'proddiff':xx*y-z*t}
 for n,m in bases.items():
  for nm,f in fun.items(): fit((step,p,n,nm),lambda k:np.c_[a,f(m,k)])
 for nm,m in [('prod',xx*y),('sq',xx**2),('triple',xx*y*z),('one',np.ones_like(xx)),('linear',xx),('sum',xx+y),('diff',xx-y)]:
  for nd,d in list(bases.items())+[('x',xx),('y',y),('z',z),('norm',np.sqrt(xx**2+y**2)),('xz',xx+z),('yz',y+z),('xt',xx+t),('yt',y+t)]:
   fit((step,p,nm,'/'+nd+'^2'),lambda k:np.c_[a,m/(1+k*d*d)])
   fit((step,p,nm,'/'+nd+'abs'),lambda k:np.c_[a,m/(1+k*abs(d))])
 for name,fn in fun.items():
  for nd,b in [('x',xx),('sum',xx+y),('diff',xx-y),('prod',xx*y)]:
   fit((step,p,'modulated',nd,name),lambda k:np.c_[a,xx*fn(b,k)])
print(*sorted(out)[:25],sep='\n')
for p in ['panel_a','panel_b','panel_c']:
 u=s[p]['u']; print(p)
 for key,v in s[p].items():
  if key=='u':continue
  f=np.c_[np.ones(len(u)),u]; w=np.linalg.lstsq(f,v,rcond=None)[0]; print(key,np.linalg.svd(w[1:],compute_uv=False),w[1:])
