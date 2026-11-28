import numpy as np,itertools
from scipy.optimize import least_squares
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def ops(s,x):
 a=s['panel_a']; good=np.max(abs(a['v']),axis=1)<4; u=a['u'][good];v=a['v'][good]
 qa=np.array([ls(np.c_[np.ones(len(u)),u[:,j],u[:,(j+1)%4],u[:,j]*u[:,(j+1)%4],u[:,j]**3],v[:,j]) for j in range(4)]).T
 def A(u):
  w=np.roll(u,-1,axis=1);return qa[0]+qa[1]*u+qa[2]*w+qa[3]*u*w+qa[4]*u**3
 z=x.copy()
 for it in range(12):
  J=np.zeros((len(x),4,4));i=np.arange(4);w=np.roll(z,-1,axis=1);J[:,i,i]=qa[1]+qa[3]*w+3*qa[4]*z*z;J[:,i,(i+1)%4]=qa[2]+qa[3]*z; z-=np.clip(np.linalg.solve(J,(A(z)-x)[...,None])[...,0],-.5,.5)
 b=s['panel_b']; c=s['panel_c']
 def trig(d,v,fn,start):
  u=d['u'];
  def f(u,g):return np.c_[aff(u),fn(u*g)]
  fit=least_squares(lambda g:(f(u,g)@ls(f(u,g),v)-v).ravel(),np.full(4,start),bounds=(.1,5)); Q=ls(f(u,fit.x),v);print('aux',fit.x,np.mean(abs(fit.fun)),Q) if __name__=='__main__' else None
  return lambda u:f(u,fit.x)@Q
 B=trig(b,b['v'],np.sin,2.85); D=trig(c,c['v2'],np.cos,1.25)
 def f(u):return np.c_[np.ones(len(u)),u[:,0]*u[:,1],u[:,0]*u[:,3],u[:,1]*u[:,2],u[:,2]*u[:,3]]
 q=ls(f(c['u']),c['v1']); C=lambda u:f(u)@q
 return z,A,B,C,D

def operators(s,x):
 z,A,B,C,D=ops(s,x)
 c=s['panel_c']; u=c['u']; v=c['v2']; f=lambda u:aff(u+.35*u*u+.065*(1-np.cos(2.5*u))); q=ls(f(u),v); D=lambda u,f=f,q=q:f(u)@q
 b=s['panel_b'];u=b['u'];v=b['v']; cand={}
 for typ in ['sinfit','odd5','odd7','odd9','cubicminus','inverse3','softplus','tanhcub','polyfull','sinplus','gauss']:
  def f(w,p):
   if typ.startswith('odd'):return aff(np.stack([w**i for i in range(1,int(typ[3:])+1,2)],2).reshape(len(w),-1))
   if typ=='polyfull':return np.c_[aff(w),w*w,w**3,w**5]
   if typ=='sinfit':return aff(w+p[1]*np.sin(p[0]*w))
   if typ=='cubicminus':return aff(w**3+p[0]*w)
   if typ=='inverse3':return aff(w+p[1]*np.sinh(np.arcsinh(p[0]*w)/3))
   if typ=='softplus':return aff(w*np.logaddexp(0,p[0]*(w*w-p[1])))
   if typ=='tanhcub':return aff(w+p[0]*np.tanh(p[1]*w)+p[2]*w**3)
   if typ=='sinplus':return aff(w+p[1]*np.sin(p[0]*w)+p[2]*w**3)
   return aff(w+p[1]*w*np.exp(-p[0]*w*w))
  p={'sinfit':[2.88,-.4],'cubicminus':[.15],'inverse3':[2,-1.5],'softplus':[2,.25],'tanhcub':[-1,1,-.08],'sinplus':[3.58,-.373,.735],'gauss':[1,-1.1]}.get(typ,[])
  if p:p=least_squares(lambda p:(f(u,p)@ls(f(u,p),v)-v).ravel(),p,max_nfev=100).x
  F=f(u,p); q=ls(F,v); Bz=f(z,p)@q; cand[typ]=Bz
  if __name__=='__main__':print('B',typ,p,np.std(F@q-v),np.max(abs(F@q-v)))
 cand['original']=B(z)
 return z,cand,C,D

def fit(F,Y,ix,jx,regs):
 a=F[ix,1:]; h=a.mean(0); sd=np.maximum(a.std(0),1e-5); a=(a-h)/sd;b=(F[jx,1:]-h)/sd; yy=Y[ix]; m=yy.mean(0); U,S,V=np.linalg.svd(a,full_matrices=False);R=U.T@(yy-m);G=b@V.T
 for reg in regs:
  wt=S/(S*S+reg); P=G@(wt[:,None]*R)+m; ht=(U**2)@(S*wt)+1/len(ix); L=(yy-m-U@((S*wt)[:,None]*R))/(1-ht[:,None]);yield reg,P,np.mean(abs(L),0)



def solve_batch(study):
    tr=study['train']; n=len(tr['x']); x=np.r_[tr['x'],study['test']['x']]
    z,bs,C,D=operators(study,x)
    d=D(z); Y=np.c_[tr['y'],tr['traj']]; ix=np.arange(n); jx=np.arange(n,len(x))
    preds=[]
    for kind,r in [('polyfull',.001),('odd5',.001)]:
        c=C(bs[kind]); F=np.c_[poly(np.tanh(1.25*np.c_[c,d]),2),c,d]
        preds.append(next(fit(F,Y,ix,jx,[r]))[1])
    P=np.mean(preds,axis=0)
    y,t=P[:,0].copy(),P[:,1:].copy()
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}
