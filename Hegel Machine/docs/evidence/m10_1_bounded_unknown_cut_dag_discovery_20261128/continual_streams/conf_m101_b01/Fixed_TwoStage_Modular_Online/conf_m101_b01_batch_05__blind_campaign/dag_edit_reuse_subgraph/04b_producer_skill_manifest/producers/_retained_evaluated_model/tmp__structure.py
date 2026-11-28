exec(open('/tmp/discover.py').read().split("for p,k in")[0])
from scipy.optimize import least_squares

def vp(fun,p,xx,yy):
 def res(p):
  X=fun(p,xx); return (X@fit(X,yy)-yy).ravel()
 return least_squares(res,p,max_nfev=250)

def rates(p,t):
 return np.column_stack([1-np.exp(-np.exp(k)*t) for k in p])
t=s['eval_times']; T=s['train']['traj'].T
for k in (1,2,3):
 o=vp(rates,np.linspace(-.6,.3,k),t,T); print('temporal rates',np.exp(o.x),'rmse',np.mean(o.fun**2)**.5,'max',max(abs(o.fun)))
for fun in (lambda p,t: np.c_[1-np.exp(-np.exp(p[0])*t),t],lambda p,t: np.c_[1-np.exp(-np.exp(p[0])*t),np.sin(np.exp(p[1])*t)],lambda p,t: np.c_[1-np.exp(-np.exp(p[0])*t),t*np.exp(-np.exp(p[1])*t)]):
 o=vp(fun,[-.2,-.5],t,T); print('other temporal',np.exp(o.x),np.mean(o.fun**2)**.5)
for name,tar in [('panel_a','v'),('panel_b','v'),('panel_c','v2')]:
 p=s[name]; U=p['u']; V=p[tar]; sel=np.argsort(r)[:45] if name=='panel_a' else np.arange(len(U)); U,V=U[sel],V[sel];
 def assess(X): return np.mean(abs(X@fit(X,V)-V),axis=0),loo(X,V)
 print('\n',name)
 for label,X in [('linear',feats(U,lin)),('pure signed',np.c_[np.ones(len(U)),U,U*abs(U)]),('diagonal',feats(U,lin+diag)),('pairs',feats(U,lin+pair)),('diag cubic',np.c_[feats(U,lin+diag),U**3]),('full quadratic',feats(U,lin+q))]:
  print(label,*[np.round(v,5) for v in assess(X)])
 if name=='panel_b':
  U0=feats(U,lin); Q=feats(U,q); D=U**2
  for i,y in enumerate(V.T):
   base=fit(U0,y); print('B channel',i,'linear',base)
   for fn in (np.tanh,np.sin,lambda x:x/(1+x*x),lambda x:x/(1+abs(x)),lambda x:x/(1+x*x)**.5):
    for add in (np.ones((len(U),1)),np.c_[np.ones(len(U)),D]):
     def rf(a):
      X=np.c_[add,fn(U0@a)]; return X@fit(X,y)-y
     best=min([least_squares(rf,g*base,max_nfev=200) for g in (.5,1,2,4)],key=lambda o:o.cost)
     print(fn.__name__,add.shape[1],round(np.mean(abs(best.fun)),5),np.round(best.x,4),np.round(fit(np.c_[add,fn(U0@best.x)],y),4))
