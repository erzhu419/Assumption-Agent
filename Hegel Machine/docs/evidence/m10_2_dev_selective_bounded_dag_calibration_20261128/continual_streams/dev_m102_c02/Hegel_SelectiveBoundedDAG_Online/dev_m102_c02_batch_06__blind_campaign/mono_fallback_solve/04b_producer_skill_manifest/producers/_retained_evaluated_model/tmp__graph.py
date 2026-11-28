exec(open('/tmp/probe.py').read().split("a=s['panel_a']")[0]);from scipy.optimize import least_squares
from itertools import product

def Amap(p):
 u,v=p['u'],p['v']; g=np.ones(len(u),bool)
 def f(x,j):
  a,b=x[:,j],x[:,(j+1)%4];return np.c_[np.ones(len(x)),a,b,a*b,a**3]
 for _ in range(3):
  m=np.array([fit(f(u[g],j),v[g,j]) for j in range(4)]); pred=np.column_stack([f(u,j)@m[j] for j in range(4)]); idx=np.argsort(np.linalg.norm(pred-v,axis=1));g[:]=False;g[idx[:45]]=True
 def direct(x):return np.column_stack([f(x,j)@m[j] for j in range(4)])
 def inv(x):
  z=x.copy(); jj=np.arange(4)
  for _ in range(14):
   J=np.zeros((len(z),4,4)); w=np.roll(z,-1,axis=1);J[:,jj,jj]=m[:,1]+m[:,3]*w+3*m[:,4]*z*z;J[:,jj,(jj+1)%4]=m[:,2]+m[:,3]*z
   z-=np.linalg.solve(J,(direct(z)-x)[...,None])[...,0]
  return z
 return direct,inv

def build(p,k):
 u,y=p['u'],p[k]; X=poly(u,1); models=[]
 def add(name,phi):
  Z=phi(u); m=fit(Z,y); r=Z@m-y; h=np.sum(Z*np.linalg.pinv(Z).T,axis=1); loo=np.sqrt(np.mean((r/np.maximum(1-h[:,None],.01))**2)); models.append((loo,name,lambda x:phi(x)@m,np.std(r)))
 for powers in [(1,2),(1,3),(1,2,3),(1,2,4),(1,3,5),(1,2,3,4),(1,2,4,6),(1,3,5,7)]:add(str(powers),lambda x,p=powers:np.column_stack([np.ones(len(x))]+[x**i for i in p]))
 funs={'tanh':lambda x,q:np.tanh(q*x),'sin':lambda x,q:np.sin(q*x),'sinh':lambda x,q:np.sinh(q*x),'rat':lambda x,q:x/(1+q*x*x),'rat2':lambda x,q:x*x/(1+q*x*x),'sqrt':lambda x,q:np.sqrt(1+q*x*x),'exp2':lambda x,q:np.exp(-q*x*x),'log2':lambda x,q:np.log1p(q*x*x),'sat':lambda x,q:x/np.sqrt(1+q*x*x),'sat3':lambda x,q:x**3/(1+q*x*x),'atan':lambda x,q:np.arctan(q*x),'asinh':lambda x,q:np.arcsinh(q*x)}
 for name,f in funs.items():
  for typ in (0,1,2,3):
   def phi(x,q,f=f,typ=typ):
    shift=q[1] if len(q)>1 else 0.; w=f(x+shift,q[0]);return np.c_[np.ones(len(x)),w] if typ in (0,2) else np.c_[np.ones(len(x)),x,w]
   def res(q):
    Z=phi(u,q);return (Z@fit(Z,y)-y).ravel()
   ini=[.6] if typ<2 else [.6,0.]; lo=[.005] if typ<2 else [.005,-2]; hi=[3.] if typ<2 else [3.,2]
   opt=least_squares(res,ini,bounds=(lo,hi),max_nfev=90);add((name,typ,np.round(opt.x,4)),lambda x,q=opt.x,phi=phi:phi(x,q))
 models.sort(key=lambda z:z[0]);print(k,[(round(l,6),n,round(r,7)) for l,n,_,r in models[:9]],flush=True);return models

if __name__=='__main__':
 a,inv=Amap(s['panel_a']); B=build(s['panel_b'],'v'); C1=build(s['panel_c'],'v1'); C2=build(s['panel_c'],'v2');
 x=np.r_[s['train']['x'],s['test']['x']]; z=np.c_[s['train']['y'],s['train']['traj']]; N=len(z)
 bases={'raw':x,'a':a(x),'ai':inv(x)}; ops={'b':B,'c':C1,'d':C2}; allx=dict(bases)
 for key,xx in bases.items():
  for k,M in ops.items():
   for j,(_,_,f,_) in enumerate(M[:2]):
    new=f(xx); allx[key+k+str(j)]=new
    for h,HH in ops.items():
     for i,(_,_,ff,_) in enumerate(HH[:1]):allx[key+k+str(j)+h+str(i)]=ff(new)
 def score(Z,reg=0):
  if reg:
   scale=np.std(Z[:52],axis=0); Z=Z/np.where(scale>1e-8,scale,1.);X=Z[:52]; beta=np.linalg.solve(X.T@X+reg*np.eye(X.shape[1]),X.T@z[:52]);
  else:beta=fit(Z[:52],z[:52]);
  r=np.abs(Z[52:N]@beta-z[52:N]);return r[:,0].mean(),r[:,1:].mean()
 rr=[]
 for k,xx in allx.items():
  for d in [1,2,3]:
   Z=poly(xx,d); rr.append((score(Z),k,str(d)))
  Z=np.c_[poly(xx,2),np.tanh(xx),xx/(1+xx*xx)];rr.append((score(Z,.03),k,'joint'))
 for kk in bases:
  for k in allx:
   if k.startswith(kk) and k!=kk:
    for d in [1,2]:
     Z=np.c_[poly(allx[kk],d),poly(allx[k],d)[:,1:]];rr.append((score(Z,.01),kk+'+'+k,str(d)))
 for j in range(2):print('leader',j,sorted(rr,key=lambda r:r[0][j])[:22]);
