exec(open('/tmp/explore.py').read().split('operators={}')[0])
p=s['panel_a']; a=I(p['u']); r=np.linalg.norm(a@ls(a,p['v'])-p['v'],axis=1); take=np.argsort(r)[:-3]; u=p['u'][take];v=p['v'][take]; a=I(u)
for k in (2,3,4):
 D=np.c_[a,*[poly(u,j) for j in range(2,k+1)]]; c=ls(D,v); e=D@c-v; print('cal degree',k,'rms',np.mean(e**2)**.5,'LOO',np.mean((e/(1-np.diag(D@np.linalg.pinv(D)))[:,None])**2)**.5); print('coefs',c if k==2 else c[:15]);print('row error',np.sort(np.linalg.norm(e,axis=1)))
for nm,f in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('rational',lambda z:z/(1+z*z)),('log',lambda z:np.log1p(z*z)),('cos',np.cos)]:
 for kind,P in [('pair',poly(u,2)),('roll',u*np.roll(u,-1,axis=1)),('sums',np.c_[[u[:,i]+u[:,j] for i,j in itertools.combinations(range(4),2)]].T),('diffs',np.c_[[u[:,i]-u[:,j] for i,j in itertools.combinations(range(4),2)]].T)]:
  def D(q):return np.c_[a,f(np.exp(q[0])*P)]
  def obj(q):d=D(q);return (d@ls(d,v)-v).ravel()
  fit=min([least_squares(obj,[np.log(g)],bounds=(-4,2)) for g in (.3,1,3)],key=lambda t:t.cost); print('cal',nm,kind,np.exp(fit.x),np.mean(fit.fun**2)**.5)
for key,out in [('b','v'),('c','v1'),('c','v2')]:
 p=s['panel_'+key];u=p['u'];v=p[out];a=I(u)
 if out=='v1':
  d=np.c_[a,poly(u,2)];print('c1 poly',ls(d,v));continue
 fs=[('erf',erf),('tanh',np.tanh),('log',lambda z:np.log1p(z*z)),('sin',np.sin)]
 for name,f in fs:
  for mode in (0,1,2):
   def D(q):
    z=u*np.exp(q[0])+np.array(q[1:] if mode else [0]); return np.c_[a,f(z)] if mode<2 else I(f(z))
   def obj(q):d=D(q);return (d@ls(d,v)-v).ravel()
   fit=min([least_squares(obj,np.r_[np.log(g),np.zeros(4) if mode else []],bounds=(-4,4),max_nfev=100) for g in (.7,1.4,3)],key=lambda t:t.cost);d=D(fit.x); print(key,out,name,mode,fit.x,'err',np.mean(fit.fun**2)**.5,'coef',ls(d,v) if np.mean(fit.fun**2)**.5<.001 else '')
