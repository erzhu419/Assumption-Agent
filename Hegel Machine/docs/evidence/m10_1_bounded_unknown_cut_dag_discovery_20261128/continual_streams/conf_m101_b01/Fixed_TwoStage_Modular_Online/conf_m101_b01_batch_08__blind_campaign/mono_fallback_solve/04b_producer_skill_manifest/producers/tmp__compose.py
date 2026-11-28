from discover import *
from scipy.optimize import least_squares
R=lambda x:np.roll(x,-1,axis=1)
cycle=lambda x:x*R(x)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def afit(s, verbose=False):
 u,v=(s['panel_a'][k] for k in ('u','v'));j=np.arange(4); prev=R(u)
 keep=np.argsort(np.linalg.norm(v-aff(u)@ls(aff(u),v),axis=1))[:45]
 def base(x,i):return np.c_[np.ones(len(x)),x[:,i],R(x)[:,i],cycle(x)[:,i]]
 candidates={ 'base':lambda x,i,f:base(x,i), 'diag':lambda x,i,f:np.c_[base(x,i),np.sin(f*x[:,i])], 'other':lambda x,i,f:np.c_[base(x,i),np.sin(f*R(x)[:,i])], 'both':lambda x,i,f:np.c_[base(x,i),np.sin(f*x[:,i]),np.cos(f*x[:,i])], 'shift':lambda x,i,f:np.c_[base(x,i),np.sin(f*(x[:,i]+R(x)[:,i]))], 'subtract':lambda x,i,f:np.c_[base(x,i),np.sin(f*(x[:,i]-R(x)[:,i]))], 'product':lambda x,i,f:np.c_[base(x,i),np.sin(f*cycle(x)[:,i])], 'symquad':lambda x,i,f:np.c_[base(x,i),x[:,i]**2,R(x)[:,i]**2,np.sin(f*x[:,i])], 'diagsecond':lambda x,i,f:np.c_[base(x,i),np.sin(f*x[:,i]),np.sin(2*f*x[:,i])], 'modulated':lambda x,i,f:np.c_[base(x,i),np.sin(f*x[:,i]),np.sin(f*x[:,i])*R(x)[:,i]], 'rational':lambda x,i,f:np.c_[base(x,i),x[:,i]/(1+f*f*x[:,i]**2)], 'cosother':lambda x,i,f:np.c_[base(x,i),np.sin(f*x[:,i]),np.cos(f*R(x)[:,i])], 'full':lambda x,i,f:np.c_[poly(x),np.sin(f*x)]}
 for _ in range(5):
  ww=[ls(base(u[keep],i),v[keep,i]) for i in j]; rr=np.column_stack([base(u,i)@ww[i]-v[:,i] for i in j]);keep=np.argsort(np.linalg.norm(rr,axis=1))[:45]
 def fit(fun,f,show=False):
  vals=[];ww=[]
  for i in j:
   a=fun(u[keep],i,f);w=ls(a,v[keep,i]);r=v[keep,i]-a@w;h=np.sum(a*np.linalg.pinv(a).T,1);vals.append(np.mean(abs(r)/(1-h)));ww.append(w)
  return (np.mean(vals),lambda x:np.column_stack([fun(x,i,f)@ww[i] for i in j]),ww)
 res=[]
 for name,fun in candidates.items():
  grid=np.linspace(.35,8,32);kk=np.argmin([fit(fun,f)[0] for f in grid]);lo,hi=grid[max(0,kk-1)],grid[min(len(grid)-1,kk+1)]; f=minimize_scalar(lambda f:fit(fun,f)[0],bounds=(lo,hi),method='bounded').x
  res.append((fit(fun,f)[0],name,f,fit(fun,f)))
 if verbose:
  print('A models',[(n,round(f,4),round(e,7)) for e,n,f,_ in sorted(res)]); print('A best coefs',sorted(res)[0][3][2]);print('A axis base coefs',ww)
 return {n:r[1] for _,n,f,r in sorted(res)}
def operators(s,verbose=False):
 u,v=(s['panel_b'][k] for k in ('u','v')); p=s['panel_c'];q=p['u'];
 def bfit(par):
  f,a=par;h=lambda x:x+a*np.sin(f*x); w=ls(aff(h(u)),v); return h,w
 pB=least_squares(lambda par:(aff(bfit(par)[0](u))@bfit(par)[1]-v).ravel(),[2.8,-.4]).x;h,b=bfit(pB);B=lambda x:aff(h(x))@b
 cc=ls(aff(cycle(q))[:,[0,1,2,3,4]],p['v1']); C=lambda x:aff(cycle(x))@cc
 def dfit(par):
  f,a,b=par;g=lambda x:x+a*x*x+b*(np.cos(f*x)-1);w=ls(aff(g(q)),p['v2']);return g,w
 pD=least_squares(lambda par:(aff(dfit(par)[0](q))@dfit(par)[1]-p['v2']).ravel(),[2.55,.36,-.06],max_nfev=300).x;g,d=dfit(pD);D=lambda x:aff(g(x))@d
 if verbose:print('operators',pB,np.mean(abs(B(u)-v)),np.mean(abs(C(q)-p['v1'])),pD,np.mean(abs(D(q)-p['v2'])))
 return {'B':B,'C':C,'D':D}
def inverse(fun,x):
 return np.array([least_squares(lambda z:(fun(z[None,:])[0]-r),r,max_nfev=70).x for r in x])
if __name__=='__main__':
 x=s['train']['x'];a=afit(s,True); op=operators(s,True);coord={'raw':x}
 for key in list(a)[:5]+['base']:
  z=inverse(a[key],x);print('inverse',key,np.max(abs(a[key](z)-x)),z.min(),z.max());coord[key]=z
 yy=np.c_[s['train']['y'],s['train']['traj']]; tr=np.arange(52);te=np.arange(52,72)
 aa=yy[:,1:];print('traj means',aa.mean(0),'time basis uncentered',np.linalg.svd(aa,full_matrices=False)[2][:3]);
 def assess(d):
  sc=np.sqrt(np.mean(d[tr]**2,axis=0));d=d/np.maximum(sc,1e-6);w=np.linalg.solve(d[tr].T@d[tr]+.04*np.eye(d.shape[1]),d[tr].T@yy[tr]); e=abs(d@w-yy);return np.r_[e[te,0].mean(),e[te,1:].mean(),e[tr,0].mean(),e[tr,1:].mean()]
 allres=[]
 for key,z in coord.items():
  zs={'':z}
  for prev in ['','B','C','D','BC','BD','CB','CD','DB','DC']:
   for k,fun in op.items():
    if k not in prev:zs[prev+k]=fun(zs[prev]);
  for name,u in zs.items():
   for typ,d in [('aff',aff(u)),('cyc',np.c_[aff(u),cycle(u)]),('poly',poly(u)),('tanh',aff(np.tanh(u))),('sin',aff(np.sin(u))),('sep',np.c_[aff(u),u*u,np.tanh(u)]),('joint',np.c_[aff(z),u]),('identjoint',np.c_[aff(z),zs['B'],zs['C'],zs['D'],u])]:allres.append((key,name,typ,assess(d)))
 for ix in range(4):
  print('ranking',ix,[(a,b,c,np.round(e,4).tolist()) for a,b,c,e in sorted(allres,key=lambda r:r[3][ix])[:12]])
