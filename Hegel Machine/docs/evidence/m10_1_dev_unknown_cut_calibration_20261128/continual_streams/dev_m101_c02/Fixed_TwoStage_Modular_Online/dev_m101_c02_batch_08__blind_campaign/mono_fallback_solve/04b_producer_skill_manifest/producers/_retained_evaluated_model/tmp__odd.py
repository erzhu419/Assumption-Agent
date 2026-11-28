from maps import *
from scipy.optimize import least_squares
p=s['panel_b'];u=p['u'];v=p['v']
def trans(x,k,n,name):
 z=np.sign(x)*abs(x)**n
 if name=='tanh':return np.tanh(k*z)
 if name=='rat':return z/(1+k*z*z)
 if name=='hyp':return z/np.sqrt(1+k*z*z)
 if name=='soft':return z/(1+k*abs(z))
 if name=='atan':return np.arctan(k*z)
 if name=='exp':return z*np.exp(-k*z*z)
 if name=='sinh':return np.arcsinh(k*z)
 if name=='direct':return np.tanh(k*x)**n

def fit(u,v,name,kind,n0=3):
 def design(x,p):
  k,n=p[:2]; z=trans(x,k,n,name); add=[]
  if kind in ('wide','cross','sat','poly','mix'): add=[x]
  if kind=='cross':add+=[x*np.roll(x,1,axis=1)]
  if kind=='sat':add+=[np.tanh(p[2]*x)]
  if kind=='poly':add+=[x**3]
  if kind=='mix':add=[np.tanh(p[2]*x)]
  return bias(np.column_stack([z]+add))
 def err(p):
  F=design(u,p);return (F@ls(F,v)-v).ravel()
 r=least_squares(err,[.8,n0,1.1],bounds=([.005,1.02,.05],[25,7,15]),max_nfev=120);p=r.x;C=ls(design(u,p),v)
 return np.sqrt(np.mean(r.fun**2)),p,C,lambda x:design(x,p)@C
if __name__=='__main__':
 from itertools import product
 rows=[]
 for n,m in product(['tanh','rat','hyp','soft','atan','exp','sinh'],['basic','wide','cross','sat','poly','mix']):
  e,p,c,f=fit(u,v,n,m); _,_,_,g=fit(u[:20],v[:20],n,m);d=np.mean(abs(g(u[20:])-v[20:]));rows.append((d,e,n,m,p,c,f))
 for d,e,n,m,p,c,f in sorted(rows,key=lambda r:r[0])[:12]:print('B',round(d,7),round(e,7),n,m,p); print(np.round(c,4)) if d==min(r[0] for r in rows) else None
 tr=s['train'];z=inv(tr['x'],calfit(s));T=tr['traj'];Y=np.c_[tr['y'],T];
 for d,e,n,m,p,c,f in sorted(rows,key=lambda r:r[0])[:8]:
  X=f(z);print(n,m,[(float(np.mean(abs((A[56:]@ls(A[:56],Y[:56])-Y[56:])[:,0]))),float(np.mean(abs((A[56:]@ls(A[:56],Y[:56])-Y[56:])[:,1:]))) ) for A in [bias(X),poly(X,2)[0]]]);
 print('times',s['eval_times']);print('ranges raw',z.min(0),z.max(0));print('ranges panel',u.min(0),u.max(0))
