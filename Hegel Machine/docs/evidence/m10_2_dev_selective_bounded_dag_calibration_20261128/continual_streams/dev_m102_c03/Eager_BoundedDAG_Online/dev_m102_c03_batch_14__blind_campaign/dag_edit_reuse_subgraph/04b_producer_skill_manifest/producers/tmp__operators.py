import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; J=np.arange(4)
def bias(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def opt(fun,p):return least_squares(fun,p,bounds=(.015,9),max_nfev=120)
shapes={'id':lambda x,a:x,'sin':lambda x,a:np.sin(a*x),'tanh':lambda x,a:np.tanh(a*x),'rat':lambda x,a:x/(1+a*x*x),'atan':lambda x,a:np.arctan(a*x),'exp':lambda x,a:np.expm1(-a*x*x),'cos':lambda x,a:np.cos(a*x)-1,'cub':lambda x,a:x**3,'asinh':lambda x,a:np.arcsinh(a*x)}
u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.linalg.norm(v-u,axis=1)<3; results=[]
for ka in shapes:
 for kb in ('sin','tanh','rat','id'):
  def f(x,p):return np.stack([np.ones_like(x),x,shapes[ka](x,p[0]),shapes[kb](np.roll(x,2,axis=1),p[1])],axis=2)
  def err(p):
   h=f(u[keep],p); return np.column_stack([h[:,j]@ls(h[:,j],v[keep,j])-v[keep,j] for j in J]).ravel()
  r=opt(err,[1,2.4]); results.append((np.std(r.fun),ka,kb,r.x))
print('A',sorted(results,key=lambda x:x[0])[:10]);
u,v=(s['panel_c'][k] for k in ('u','v2')); results=[]
for lag in (1,-1,2):
 for pair in ('prod','sum','diff','dir','sqdiff'):
  for name,fn in shapes.items():
   for shared in (False,True):
    def f(x,p):
     w=np.roll(x,lag,axis=1); h={'prod':x*w,'sum':x+w,'diff':x-w,'dir':x*w*w,'sqdiff':x*x-w*w}[pair]; d=fn(h,p[0]); return bias(x+p[1]*d) if shared else bias(np.c_[x,d]);
    def err(p):
     h=f(u,p); b=ls(h,v); e=h@b-v; return e.ravel()
    for p0 in ([.8,.25],[2.15,.25],[4,.25]) if name in ('sin','cos') else ([1,.25],):
     r=least_squares(err,p0,bounds=([.01,-3],[9,3]),max_nfev=100); h=f(u,r.x); lev=np.sum(h*np.linalg.pinv(h).T,axis=1); results.append((np.std(r.fun),np.sqrt(np.mean((r.fun.reshape(v.shape)/(1-lev[:,None]))**2)),lag,pair,name,shared,r.x))
print('C2');
for row in sorted(results,key=lambda x:x[1])[:22]:print(row)
print('counts',[(k,{t:np.shape(x) for t,x in z.items()}) for k,z in s.items() if isinstance(z,dict)],'outliers',np.flatnonzero(~keep));
