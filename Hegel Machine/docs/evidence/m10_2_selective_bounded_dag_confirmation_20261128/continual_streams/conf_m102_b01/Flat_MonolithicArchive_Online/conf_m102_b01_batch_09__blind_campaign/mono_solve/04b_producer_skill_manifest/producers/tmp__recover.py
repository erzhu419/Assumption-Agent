from explore import s,t,x,y,P,aff,ls,poly
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root
A=s['panel_a']; U,V=A['u'],A['v']; O=np.roll(U,2,axis=1)
fs={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'asinh':np.arcsinh,'soft':lambda x:x/np.sqrt(1+x*x),'rat':lambda x:x/(1+x*x)}
def calib(name1,name2,kind):
 f,g=fs[name1],fs[name2]; keep=np.full((48,4),True)
 def basis(u,p,j):
  q,r=u[:,j],u[:,(j+2)%4]; extra=[r] if kind else [];return aff(np.c_[q,*extra,f(p[0]*q),g(p[1]*r)])
 def ev(p,get=False):
  ds=[basis(U,p,j) for j in range(4)]; ws=[ls(d[k],v[k]) for d,k,v in zip(ds,keep.T,V.T)]; err=np.array([d@w-v for d,w,v in zip(ds,ws,V.T)]).T;return (ws,err) if get else err[keep]
 p=np.array([1.,2.15]);
 for it in range(8):
  if it==0:
   _,e=ev(p,True); keep=np.abs(e)<=np.sort(abs(e),axis=0)[38]
  p=least_squares(ev,p,bounds=(.15,6),max_nfev=80).x; ws,e=ev(p,True); keep=np.abs(e)<=np.sort(abs(e),axis=0)[39]
 score=np.sqrt(np.mean(e[keep]**2)); return score,p,np.array(ws),basis,e

def getcoords(xx, verbose=False):
 rr=sorted([(calib(a,b,k),a,b,k) for a in fs for b in fs for k in [0,1]],key=lambda r:r[0][0]);
 if verbose:
  for (e,p,ws,_,_),a,b,k in rr[:12]:print('A',e,a,b,k,p,ws.round(5))
 (err,p,ws,f,e),a,b,k=rr[0]; z=np.array([root(lambda q:np.array([f(q[None],p,j)[0]@ws[j] for j in range(4)])-r,r,tol=1e-10).x for r in xx]);
 b=s['panel_b'];w=aff(z)@ls(aff(b['u']),b['v']); c=s['panel_c'];u=c['u']; res={}
 for key in ['v1','v2']:
  v=c[key]; scores=[]
  for nm,g in {**fs,'sq':lambda x:x*x,'cos':np.cos,'exp':lambda x:-np.expm1(-x*x),'log':lambda x:np.log1p(x*x)}.items():
   for kind in [0,1,2]:
    def basis(x,p):return aff(np.c_[x,g(p*x)]) if kind==0 else (aff(g(p*x)) if kind==1 else aff(np.c_[x,g(p*x),x*x]))
    def loss(p):
     d=basis(u,p); pi=np.linalg.pinv(d); r=(d@(pi@v)-v)/(1-np.sum(d*pi.T,axis=1))[:,None];return np.mean(r*r)**.5
    p=minimize_scalar(loss,bounds=(.08,6),method='bounded').x;scores.append((loss(p),nm,kind,p))
  scores.sort();
  if verbose:print(key,scores[:12]);
  _,nm,kind,p=scores[0];g={**fs,'sq':lambda x:x*x,'cos':np.cos,'exp':lambda x:-np.expm1(-x*x),'log':lambda x:np.log1p(x*x)}[nm];
  d=lambda x,g=g,p=p,kind=kind:aff(np.c_[x,g(p*x)]) if kind==0 else (aff(g(p*x)) if kind==1 else aff(np.c_[x,g(p*x),x*x])); cf=ls(d(u),v); res[key]=lambda x,d=d,cf=cf:d(x)@cf
  if verbose:print('C coef',cf.round(4));
 return z,w,res
if __name__=='__main__':
 z,w,f=getcoords(x,True); print('ranges',z.min(0),z.max(0),w.min(0),w.max(0)); Z={'x':x,'z':z,'w':w};
 for n,q in list(Z.items()):
  Z[n+'1']=f['v1'](q);Z[n+'2']=f['v2'](q)
 np.savez('/tmp/coords.npz',**Z); T=np.c_[y,P];
 for n,q in Z.items():
  for nm,F in [('aff',q),('poly',poly(q,2)),('pair',np.c_[q,q*np.roll(z,1,axis=1),q*np.roll(z,-1,axis=1)]),('add',np.c_[q,w]),('dual',np.c_[q,Z.get(n[:-1]+('2' if n.endswith('1') else '1'),z)])]:
   d=aff(F); r=d@ls(d[:52],T[:52])-T; print(n,nm,np.round([abs(r[:52,0]).mean(),abs(r[52:,0]).mean(),abs(r[:52,1:]).mean(),abs(r[52:,1:]).mean()],4))
