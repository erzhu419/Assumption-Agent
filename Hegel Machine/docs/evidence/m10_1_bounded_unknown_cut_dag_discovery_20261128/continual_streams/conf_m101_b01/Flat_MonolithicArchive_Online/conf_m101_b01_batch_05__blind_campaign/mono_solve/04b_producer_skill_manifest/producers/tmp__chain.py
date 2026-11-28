from probe import *
from scipy.optimize import root
from itertools import product

def ap(u,kind,k=0):
 w=np.roll(u,-1,axis=1)
 if kind=='cyclic':return np.stack([np.ones_like(u),u,w,u*w],-1)
 if kind=='extra':return np.stack([np.ones_like(u),u,w,u*w,u*u,w*w,u*u*w,u*w*w,u**3,w**3],-1)
 g=funcs[kind]
 return np.stack([np.ones_like(u),u,w,u*w,g(w,k),u*g(w,k)],-1)

def cal():
 u=s['panel_a']['u']; v=s['panel_a']['v']; d=ap(u,'cyclic'); w=fit(d[:,0],v[:,0]); mask=np.ones(len(u),bool)
 for _ in range(12):
  b=np.column_stack([fit(d[mask,j],v[mask,j]) for j in range(4)]); e=np.linalg.norm(np.einsum('nki,ik->nk',d,b)-v,axis=1); mask=np.isin(np.arange(len(u)),np.argsort(e)[:38]);
 print('A cyclic',np.round(b,5),'residual quantiles',np.quantile(e,[0,.25,.5,.75,.8,.9,1])); good=e<.40
 def fit_a(kind,k):
  d=ap(u,kind,k); b=np.column_stack([fit(d[good,j],v[good,j]) for j in range(4)]); e=np.einsum('nki,ik->nk',d,b)-v; return np.sqrt(np.mean(e[good]**2)),b
 print('A polynomial',fit_a('extra',0));
 for kind in ['sin','tanh','rat','erf','atan']:
  o=minimize_scalar(lambda k:fit_a(kind,k)[0],bounds=(.05,4),method='bounded'); print('A',kind,o.fun,o.x,np.round(fit_a(kind,o.x)[1],4))
 def inverse(x,kind='cyclic',k=0):
  b=fit_a(kind,k)[1]; return np.array([root(lambda a:np.einsum('ki,ik->k',ap(a[None],kind,k)[0],b)-row,row).x for row in x])
 return inverse

def bfit(panel):
 u,v=panel['u'],panel['v']; models=[]
 ff=dict(funcs, **{'sqrat':lambda x,k: x/(1+k*x*x)**2,'gaussodd':lambda x,k:x*np.exp(-k*x*x),'sinc':lambda x,k:np.sin(k*x)/k})
 for name,f in ff.items():
  def get(x,k):return np.c_[aff(x),f(x,k)]
  def obj(k):return err(get(u,k),v)
  grid=np.geomspace(.03,10,55); a=np.argmin([obj(k) for k in grid]); o=minimize_scalar(obj,bounds=(grid[max(0,a-1)],grid[min(len(grid)-1,a+1)]),method='bounded'); k=o.x
  b=fit(get(u,k),v); models.append((o.fun,name,k,lambda x,f=f,k=k,b=b:np.c_[aff(x),f(x,k)]@b));
  if name in ['erf','rat','sin','gaussodd']:print('B',name,o.fun,k,np.round(b,4))
 def h(x,k,p):return x/(1+k*x*x)**p
 def d(x,k,p):return np.c_[aff(x),h(x,k,p)]
 def resid(a):
  z=d(u,*np.exp(a));return (z@fit(z,v)-v).ravel()
 o=least_squares(resid,[0,0],bounds=(-5,5)); k,p=np.exp(o.x); b=fit(d(u,k,p),v); print('B rational shape',k,p,np.std(o.fun)); models.append((np.std(o.fun),'rational',k,lambda x:d(x,k,p)@b))
 return sorted(models,key=lambda z:z[0])

def cfit(panel):
 u=panel['u']; out={}
 for name,f,g in [('C1',lambda x:np.c_[aff(x),x[:,l]*x[:,r]],'v1'),('C2',lambda x:np.c_[aff(x),np.exp(-.3*x*x)],'v2')]:
  b=fit(f(u),panel[g]); print(name,np.round(b,4));out[name]=lambda x,f=f,b=b:f(x)@b
 return out

if __name__=='__main__':
 inverse=cal(); bm=bfit(s['panel_b']); cm=cfit(s['panel_c']); tr=s['train']; x,y,t=tr['x'],tr['y'],tr['traj']; n=52
 out=[]
 for an,a in [('raw',x),('cal',inverse(x)),('calextra',inverse(x,'extra'))]:
  for bn,b in [('noB',a)]+[(z[1],z[3](a)) for z in bm]:
   for cn,c in [('noC',b)]+[(k,f(b)) for k,f in cm.items()]:
    for dn,d in [('aff',aff(c)),('quad',np.c_[aff(c),c[:,i]*c[:,j]])]:
     pp=d[n:]@fit(d[:n],np.c_[y,t][:n]); e=np.mean(abs(pp-np.c_[y,t][n:]),axis=0); out.append((e[0],e[1:].mean(),an,bn,cn,dn))
 print('best by y',sorted(out)[:18]);print('best by trajectory',sorted(out,key=lambda z:z[1])[:22])
