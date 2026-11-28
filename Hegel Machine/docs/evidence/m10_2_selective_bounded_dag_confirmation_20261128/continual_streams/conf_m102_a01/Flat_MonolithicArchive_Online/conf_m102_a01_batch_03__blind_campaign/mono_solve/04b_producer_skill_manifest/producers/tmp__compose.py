import sys
from gp import *
from scipy.optimize import least_squares
from scipy.special import expit
np.set_printoptions(precision=4,suppress=True)
print('y min/max/std',y.min(),y.max(),y.std(),'summary',s['train']['summary']); print('B',B)
C=s['panel_c']; U=C['u']; funcs=[lambda w:w/(1+.84*abs(w)),lambda w:np.sqrt(1+1.1*w*w)]; Cs=[np.linalg.lstsq(np.c_[A(U),f(U)],C[k],rcond=None)[0] for f,k in zip(funcs,['v1','v2'])]; print('C',Cs)
I=np.array(list(comb(range(4),2))); quad=lambda w:np.c_[A(w),np.prod(w[:,I],axis=2)]
Y=np.array([y[te] for tr,te in folds]); arch=np.load('/tmp/gpstudy.npz'); base=arch[str(('arch',0.))]; candidates={}
def record(n,p):
 p=np.array(p); e=np.mean(abs(p-Y),1); eb=np.mean(abs((p+base)*.5-Y),1);candidates[n]=(e,eb,p)
def fitmake(fun,par=(),invert=None,name='',reg=0):
 out=[]; costs=[]
 for tr,te in folds:
  v=y[tr] if invert is None else invert[0](y[tr]);
  def solve(p):
   F=fun(p); G=F[tr]; c=np.linalg.lstsq(G,v,rcond=1e-8)[0]; return F,G@c-v,c
  p=np.array(par,float)
  if p.size:p=least_squares(lambda p:np.r_[solve(p)[1],reg*p],p,max_nfev=45).x
  F,r,c=solve(p); a=F[te]@c;out.append(a if invert is None else invert[1](a));costs.append((np.sqrt(np.mean(r*r)),p))
 record(name,out)
 return costs
for i,w in enumerate(Ws[:5]):
 q=quad(w); H=w[:,:,None]*w[:,None,:]; pairs=[w[:,a]+sgn*w[:,b] for a in range(4) for b in range(a+1,4) for sgn in [-1,1]]
 for j,f in enumerate([lambda w:w,lambda w:np.tanh(1.25*w),lambda w:np.sin(1.25*w),lambda w:w/(1+.22*abs(w)),lambda w:w/(1+.22*w*w),lambda w:w/np.sqrt(1+.22*w*w),lambda w:np.arcsinh(w),lambda w:w+.22*w**2,lambda w:w/(1+.22*w),lambda w:w+.22*np.tanh(1.25*w),lambda w:w-.22*np.tanh(1.25*w)]):
  fw=f(w);
  if np.all(np.isfinite(fw)):fitmake(lambda _:quad(fw),name=('warp',i,j))
  fd=f(w[:,None,:]-w[:,:,None]); fd=fd[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]]
  fitmake(lambda _:np.c_[q,fd],name=('pairwarp',i,j))
 for k,Cj in enumerate(Cs):
  f=funcs[k]; act=np.c_[A(w),f(w)]@Cj
  for a in [-1.,-.22,.22,1.]:fitmake(lambda _:quad(w+a*act),name=('cblend',i,k,a))
  fitmake(lambda p:quad(w+p[0]*act),[.22],name=('cfit',i,k),reg=.012)
  fitmake(lambda p:quad(w+act*p),[.22]*4,name=('cfitv',i,k),reg=.025)
 for a in [.08,.22,.5,1.25]:
  for k in [0,1]:
   r=(w*w).sum(1) if k==0 else abs(w).sum(1)
   for typ in [0,1]:fitmake(lambda _:q/(1+a*r[:,None])**(1 if typ==0 else .5),name=('global',i,a,k,typ))
 for fn in [lambda v:np.arcsinh(v),lambda v:v**3,lambda v:np.arctanh(np.clip(v,-.999,.999))]:pass
 for a in [.22,.5,1.,2.]:
  fitmake(lambda _:q,invert=(lambda v:np.arcsinh(a*v),lambda v:np.sinh(np.clip(v,-4,4))/a),name=('arcsinh',a,i))
  if abs(y).max()*a<1:fitmake(lambda _:q,invert=(lambda v:np.arctanh(a*v),lambda v:np.tanh(v)/a),name=('atanh',a,i))
 # common nonlinear transformation with train-fitted scale and shift; direct compositions
 for j,f in enumerate([lambda w:w/(1+abs(w)),lambda w:np.tanh(w),lambda w:np.sin(w),lambda w:w/(1+w*w),lambda w:np.sqrt(1+w*w),lambda w:w*w]):
  for k in [-1,0,1]:
   if k==-1:fun=lambda p:quad(w+p[0]*f(p[1]*w+p[2]))
   else:
    c=Cs[k]; L=funcs[k]; fun=lambda p:quad(w+p[0]*f(p[1]*(np.c_[A(w),L(w)]@c)+p[2]))
   fitmake(fun,[.22,1.25,0],name=('profwarp',i,j,k),reg=.018)
for name,(e,eb,p) in sorted(candidates.items(),key=lambda kv:kv[1][0].mean())[:12]:print('average',name,e.round(4),e.mean(),eb.round(4))
for name,(e,eb,p) in sorted(candidates.items(),key=lambda kv:kv[1][0][0])[:12]:print('gate',name,e.round(4),e.mean(),eb.round(4))
keys=[('arch',0.),('iso',0.),('ard',0.),('full',1.),('full',3.),('all',0.),('all',1.),('all',3.)];
for ids in [[0,4,5],[0,1,4,5],[0,1,4,6,7],[0,1,3,4,5,6,7],[0,1,2,3,4,5,6,7]]:
 e=np.mean(abs(np.mean([arch[str(keys[i])] for i in ids],axis=0)-Y),axis=1);print('gp avg',ids,e.round(5),e.mean(),max(e))
