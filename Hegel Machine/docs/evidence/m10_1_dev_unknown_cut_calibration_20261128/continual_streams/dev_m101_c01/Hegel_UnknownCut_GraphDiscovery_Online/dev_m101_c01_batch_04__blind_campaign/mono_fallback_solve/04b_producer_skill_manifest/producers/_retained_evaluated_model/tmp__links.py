from operators import *
from scipy.optimize import least_squares
from itertools import combinations
np.set_printoptions(precision=4,suppress=True)
_,_,vt=np.linalg.svd(z,full_matrices=False); Y=np.c_[y,z@vt[0]*vt[0,-1]]
print('output ranges',np.quantile(Y,[0,.05,.5,.95,1],axis=0)); print('spearman y vs amp',np.corrcoef(np.argsort(np.argsort(Y,axis=0),axis=0).T))
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def pol(a,d):return PolynomialFeatures(d).fit_transform(a)
B=aff(x)@bb; Br=aff(raw)@bb
for name,U in [('A',x),('B',B),('Br',Br)]:
 F=pol(U,3); power=PolynomialFeatures(3).fit(U).powers_; co=fit(F,Y); print(name,'cubic coefficients',[(tuple(q),list(np.round(c,3))) for q,c in zip(power,co) if sum(q)==3]);
for tgt in range(2):
 yy=Y[:,tgt]; results=[]
 for name,U in [('A',x),('B',B),('Br',Br),('BC1',c1map(B)),('BC2',c2map(B))]:
  F=pol(U,2); H=pol(x,2)
  for fkey,F0 in [('quadratic',F),('Cmix',np.c_[H,U,c1map(B),c2map(B)]),('Ccross',np.c_[H,cross(B,c2map(B))]),('cubic',pol(U,3))]:
   if fkey in ['Cmix','Ccross'] and name!='B':continue
   for kind in ['tanh','asinh','rational_norm','rational_axis','rational_index','quadratic_index']:
    if fkey=='cubic' and kind not in ['rational_norm','rational_axis']:continue
    ee=[]; rr=[]; params=[]
    for tr,te in folds:
     M=F0[tr]; N=F0[te]; yt=yy[tr]; V=U[tr]; Vt=U[te];
     if kind in ['tanh','asinh']:
      cen=yt.mean(); lim=max(abs(yt-cen));
      def design(p):
       off=p[0]+cen; w=p[1]; q=(yt-off)/w
       if kind=='tanh': return off,w,np.arctanh(np.clip(q,-.998,.998))
       return off,w,np.arcsinh(q)
      def pred(p):
       off,w,q=design(p); c=fit(M,q); fn=np.tanh if kind=='tanh' else np.sinh; return off+w*fn(np.clip(M@c,-10,10)),off+w*fn(np.clip(N@c,-10,10))
      bounds=([-lim*.5,lim*.95 if kind=='tanh' else .03],[lim*.5,lim*20]); ini=[0,lim*1.4 if kind=='tanh' else lim*.6]
     else:
      def den(p,u):
       if kind=='rational_norm':return 1+p[0]*np.sum(u*u,axis=1)
       if kind=='rational_axis':return 1+(u*u)@p
       l=p[1]+u@p[2:]; return 1+p[0]*l*l
      def pred(p):
       if kind=='quadratic_index':
        l=p[1]+V@p[2:]; lt=p[1]+Vt@p[2:]; a=p[0]*l**4; at=p[0]*lt**4; c=fit(M,yt-a);return M@c+a,N@c+at
       d=den(p,V); dt=den(p,Vt); c=fit(M/d[:,None],yt);return (M@c)/d,(N@c)/dt
      if kind=='rational_norm': ini=[.2];bounds=([0],[8])
      elif kind=='rational_axis': ini=np.full(4,.2);bounds=(np.zeros(4),np.full(4,8))
      else:ini=[.2,.01,.4,.3,.2,.1];bounds=([-5 if kind=='quadratic_index' else 0,-2,-2,-2,-2,-2],[8,2,2,2,2,2]);
     r=least_squares(lambda p:pred(p)[0]-yt,ini,bounds=bounds,max_nfev=45); a,b0=pred(r.x); ee.append(np.mean(abs(b0-yy[te])));rr.append(np.std(a-yt));params.append(r.x)
    results.append((np.mean(ee),max(ee),ee[0],np.mean(rr),name,fkey,kind,params[0]))
 print('BEST LINKS',tgt); [print(v) for v in sorted(results)[:12]]
