from graph import *
from scipy.optimize import least_squares
from scipy.linalg import svd
A,AI=Amap(s['panel_a']); x=AI(s['train']['x']); t=s['eval_times']; y=s['train']['y']; tr=s['train']['traj']; D,U,V=np.linalg.svd(tr,full_matrices=False);print('t',t,'singular',U,'temporal basis',V[:5], 'summary',s['train']['summary'],'tele',s['telemetry_vec']); print('calibrated sample:'); print(np.c_[x,y,tr][:15]);print('range',x.min(0),x.max(0),'inverse residual',np.max(abs(A(x)-s['train']['x'])));print('y from traj',np.std(poly(tr,1)@fit(poly(tr,1),y)-y));
for n in [1,2,3,4]:
 def E(p):return -np.expm1(-np.exp(p[:,None])*t)
 def loss(p):
  v=E(p);return (tr@np.linalg.pinv(v)@v-tr).ravel()
 p=least_squares(loss,np.linspace(-2,1.5,n),bounds=(-6,5),max_nfev=500).x
 print('modes',n,np.exp(p),np.std(loss(p)))
for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[k]['u'],s[k][j]; X=poly(u,1); print(k,j,'ranges',u.min(0),u.max(0));
 F={'sat3':lambda u,q:u**3/(1+q*u*u),'asin':lambda u,q:np.arcsinh(q*u),'tanh':lambda u,q:np.tanh(q*u),'atan':lambda u,q:np.arctan(q*u),'log2':lambda u,q:np.log1p(q*u*u),'sqrt':lambda u,q:np.sqrt(1+q*u*u),'rat2':lambda u,q:u*u/(1+q*u*u),'sin':lambda u,q:np.sin(q*u),'softsign':lambda u,q:u/(1+q*abs(u)),'cubeabs':lambda u,q:u**3/(1+q*abs(u)),'logcosh':lambda u,q:np.logaddexp(q*u,-q*u),'cubic':lambda u,q:u**3/(1+q*u**4),'sigmoid':lambda u,q:u/(1+q*u**4)}
 for name,f in F.items():
  best=(1e9,)
  for rank in [0,1,2]:
   def calc(p):
    w=f(u,np.exp(p[:4])); Z=np.c_[X,w] if rank!=1 else np.c_[np.ones(len(u)),u+p[4:]*w]; M=fit(Z,v);r=Z@M-v;h=np.sum(Z*np.linalg.pinv(Z).T,axis=1);return r,M,np.std(r/np.maximum(1-h[:,None],.02))
   p=least_squares(lambda p:calc(p)[0].ravel(),np.r_[np.zeros(4),np.full(4,.5)] if rank==1 else np.zeros(4),bounds=(-5,5),max_nfev=120).x
   r,m,cv=calc(p)
   if cv<best[0]:best=(cv,np.std(r),rank,p,m)
  print(name,best[:4]);
  if best[0]<.0005:print(best[-1])
