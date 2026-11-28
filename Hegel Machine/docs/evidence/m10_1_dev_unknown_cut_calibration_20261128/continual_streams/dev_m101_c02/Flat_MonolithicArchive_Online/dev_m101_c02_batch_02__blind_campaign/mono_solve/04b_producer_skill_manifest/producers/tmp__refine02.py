exec(open('/tmp/explore02.py').read().split('p=least_squares')[0])
rng=np.random.default_rng(224); D=np.c_[A(z),z[:,0]*z[:,2]+z[:,1]*z[:,3]]; h0=D@np.linalg.lstsq(D,y,rcond=None)[0]; splits=[va]+[rng.permutation(len(y))[:16] for _ in range(6)]+[np.argsort(h0)[np.r_[0:8,-8:0]]]; print('times',t); print('panel A flagged',np.flatnonzero(~keep));
for name,fn in [('atan',np.arctan),('tanh',np.tanh),('rational',lambda h:h/(1+h*h)),('gaussian',lambda h:h*np.exp(-h*h)),('sin',np.sin),('erf',__import__('scipy.special',fromlist=['erf']).erf),('asinh',np.arcsinh),('softsign',lambda h:h/np.sqrt(1+h*h))]:
 es=[]; tails=[]; fitted=None
 for vi in splits+[np.arange(len(y))]:
  ti=np.setdiff1d(np.arange(len(y)),vi) if len(vi)<len(y) else vi; init=np.linalg.lstsq(D[ti],y[ti],rcond=None)[0];
  def model(p,idx):
   h=D[idx]@p; return np.c_[np.ones(len(h)),h,fn(h)]
  def cf(p):return np.linalg.lstsq(model(p,ti),y[ti],rcond=None)[0]
  fits=[least_squares(lambda p:model(p,ti)@cf(p)-y[ti],init*v,max_nfev=150) for v in (.35,.7,1.15,1.6)]; fit=min(fits,key=lambda o:o.cost).x; e=np.abs(model(fit,vi)@cf(fit)-y[vi]); es.append(e.mean()); tails.append(e.max()); fitted=(fit,cf(fit))
 print('scalar',name,np.round(es,5), 'tail max',round(tails[-2],5),'params',*[p.round(4) for p in fitted]);
print('trajectory normalized shapes', (np.linalg.lstsq(z,T,rcond=None)[0]/np.linalg.lstsq(z,T,rcond=None)[0][:,-1:]).round(6));
fns={'exp':lambda r:-np.expm1(-r*t),'log':lambda r:np.log1p(r*t),'rat':lambda r:t/(1+r*t),'sqrt':lambda r:np.sqrt(1+r*t)-1,'tanh':lambda r:np.tanh(r*t),'linear':lambda r:t};
for n,m in itertools.combinations_with_replacement(fns,2):
 if n==m=='linear':continue
 es=[]
 for vi in splits:
  ti=np.setdiff1d(np.arange(len(y)),vi); co=np.linalg.lstsq(z[ti],T[ti],rcond=None)[0]
  def predict(p):
   B=np.array([fns[n](p[0]),fns[m](p[1])]); return co@np.linalg.pinv(B)@B
  fit=min([least_squares(lambda p:(z[ti]@(predict(p)-co)).ravel(),v,bounds=(.0002,20),max_nfev=85) for v in ([.5,1.5],[1.5,.5])],key=lambda o:o.cost);
  es.append(np.abs(z[vi]@predict(fit.x)-T[vi]).mean())
 print('traj',n,m,np.round(es,5),'rates',fit.x.round(4))
