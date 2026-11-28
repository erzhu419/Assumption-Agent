exec(open('/tmp/operators.py').read().split('ops={}')[0])
from scipy.special import erf, expit
u=s['panel_b']['u']; v=s['panel_b']['v']; rng=np.random.default_rng(710)
ids=[(np.arange(72),52),(np.arange(72),56),*[(rng.permutation(72),56) for _ in range(8)],*[(np.argsort(abs(z[:,j])),52) for j in range(4)]]
funcs={
 'erf':(erf,[.68,1.45]),'tanh':(np.tanh,[.75,1.43]),'sin':(np.sin,[.4,2.84]),
 'exp':(lambda h:h*np.exp(-h*h),[1.09,1.01]),
 'rational2':(lambda h:h/(1+h*h)**2,[1.1,.7]),
 'logistic':(lambda h:h*expit(1-h*h),[1.5,1.3]),
 'gausscdf':(lambda h:h*(1-erf(h*h)),[1.1,.8]),
 'sinc':(lambda h:h*np.sinc(h)**2,[1.75,.5]),
}
cy=[0,1,2,3,4,7,11]
ct={'lin':list(range(5)), 'small':[0,1,2,3,4,7,8,11], 'quad':list(range(15))}

def warp(x,p,fun):return lin(x-p[0]*fun(p[1]*x))

def panel(fun,q):
 def res(p):
  D=warp(u,p,fun);return (D@fit(D,v)-v).ravel()
 o=least_squares(res,q,bounds=([.02,.15],[4.,4.])); return o.x,fit(warp(u,o.x,fun),v),np.std(o.fun)

def basis(T):
 def B(q):return -np.expm1(-t[:,None]*q)
 def res(q):
  b=B(q);return (b@fit(b,T.T)-T.T).ravel()
 o=least_squares(res,[1.6,.3],bounds=([.58,.01],[5.,1.2])); return np.linalg.qr(B(o.x))[0],o.x,np.std(o.fun)

def evaluate(kind,mode,w,bmix):
 fun,q=funcs[kind]; p,C,pr=panel(fun,q); results=[]
 for ix,n in ids:
  i,j=ix[:n],ix[n:]; Q,rt,terr=basis(Y[i,1:]); target=Y[i,1:]@Q
  def design(zi,pp,cc):
   b=warp(zi,pp,fun)@cc; g=poly(b,2);return g[:,cy],g[:,ct[mode]]
  def score(par,ret=False):
   pp=par[:2]; cc=par[2:].reshape(5,4) if bmix else C
   if not bmix:
    au=warp(u,pp,fun);cc=fit(au,v)
   A,B=design(z[i],pp,cc); ca=fit(A,Y[i,0]); cb=fit(B,target)
   if ret:
    Aj,Bj=design(z[j],pp,cc);return np.c_[Aj@ca,Bj@cb@Q.T]
   return np.r_[(warp(u,pp,fun)@cc-v).ravel(),w*(A@ca-Y[i,0]),w*(B@cb-target).ravel()/np.sqrt(8)]
  p0=np.r_[p,C.ravel()] if bmix else p
  if w:
   box=([.02,.15]+([-np.inf]*20 if bmix else []),[4.,4.]+([np.inf]*20 if bmix else [])); pp=least_squares(score,p0,bounds=box,max_nfev=45).x
  else: pp=p0
  E=np.abs(score(pp,True)-Y[j]);results.append([E[:,0].mean(),E[:,1:].mean(),E[:,-1].mean()]);
 R=np.array(results); print(kind,mode,w,bmix,'panel',np.round([*p,pr],5),'metrics',np.round(np.r_[R[0],R[1],R[2:10].max(0),R[10:].max(0)],5),flush=True)
for kind in funcs:
 for mode in ('lin','small','quad'):
  evaluate(kind,mode,0,False)
for k in ('erf','tanh','exp','logistic','sin'):
 for w,mix in ((.65,False),(.65,True),(1.65,True)):
  evaluate(k,'small',w,mix)
Q,rt,terr=basis(Y[:,1:]);print('rates, temporal reconstruction rmse',rt,terr)
