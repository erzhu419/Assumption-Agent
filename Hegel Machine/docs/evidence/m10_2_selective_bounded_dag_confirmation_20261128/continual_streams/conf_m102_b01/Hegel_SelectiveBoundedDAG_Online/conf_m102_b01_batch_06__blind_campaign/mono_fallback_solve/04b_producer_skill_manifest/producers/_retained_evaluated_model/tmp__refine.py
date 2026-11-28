from graph import *
from sklearn.linear_model import lars_path
from scipy.special import erf
from scipy.optimize import minimize_scalar

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

def features(z,b,c,d,gain=.22):
 e=np.c_[c,d]
 yield 'plain',poly(e,2)
 yield 'eraw',np.c_[poly(e,2),b,z]
 for name,fn in [('id',lambda a:a),('tanh',lambda a:np.tanh(1.25*a)),('sin',lambda a:np.sin(1.25*a)),('asinh',lambda a:np.arcsinh(1.25*a)),('atan',lambda a:np.arctan(1.25*a)),('rat',lambda a:a/(1+gain*a*a)),('cubr',lambda a:np.sinh(np.arcsinh(1.25*a)/3)),('ex',lambda a:np.expm1(np.clip(a,-4,4)))]:
  p=fn(c); q=fn(d)
  yield name+'both',poly(np.c_[p,q],2)
  yield name+'c',poly(np.c_[p,d],2)
  yield name+'d',poly(np.c_[c,q],2)
  yield name+'sum',np.c_[poly(c+gain*d,2),d]
  yield name+'full',np.c_[poly(np.c_[p,q],2),c,d]
  for op,f in [('add',c+q),('mult',c*(1+gain*q)),('gate',c/(1+gain*d*d)),('shiftgate',(c+d)/(1+gain*d*d)),('cross',c*np.roll(d,1,axis=1))]:
   yield name+op,np.c_[poly(f,2),d,c]
 yield 'inter',np.c_[poly(e,2),np.column_stack([c[:,i]*d[:,j]**2 for i in range(4) for j in range(4)])]
 yield 'restricted',np.c_[poly(c,2),poly(d,2)[:,1:],c*d,c*np.roll(d,1,axis=1),c*np.roll(d,-1,axis=1)]

def fit(F,Y,ix,jx,regs):
 a=F[ix,1:]; h=a.mean(0); sd=np.maximum(a.std(0),1e-5); a=(a-h)/sd;b=(F[jx,1:]-h)/sd; yy=Y[ix]; m=yy.mean(0); U,S,V=np.linalg.svd(a,full_matrices=False);R=U.T@(yy-m);G=b@V.T
 for reg in regs:
  wt=S/(S*S+reg); P=G@(wt[:,None]*R)+m; ht=(U**2)@(S*wt)+1/len(ix); L=(yy-m-U@((S*wt)[:,None]*R))/(1-ht[:,None]);yield reg,P,np.mean(abs(L),0)

if __name__=='__main__':
 tr=s['train'];Y=np.c_[tr['y'],tr['traj']];z,bs,C,D=operators(s,tr['x']); ix=np.arange(52);jx=np.arange(52,72);rows=[]
 for typ,b in bs.items():
  for dc,d in [('z',D(z)),('b',D(b)),('z+b',D(z)+D(b)),('z-b',D(z)-D(b))]:
   for key,F in features(z,b,C(b),d):
    for reg,P,L in fit(F,Y,ix,jx,[.001,.01,.05,.2,1.]):
     v=abs(P-Y[jx]).mean(0);rows.append((L[0]+L[1:].mean(),v[0],v[1:].mean(),typ,dc,key,reg))
 for col in (0,1,2):print('BEST',col,*sorted(rows,key=lambda r:r[col])[:14],sep='\n')
