exec(open('/tmp/discover.py').read().split('A=old._calibration')[0]);from scipy.optimize import least_squares
A=old._calibration(s['panel_a']);z=old._decode(x,A);pb=s['panel_b']; B=fit(poly(pb['u'],1),pb['v']);w=poly(z,1)@B
pc=s['panel_c']; U=pc['u']; Ub=poly(U,1)
for field in ('v1','v2'):
 V=pc[field]
 print('\n',field)
 for label, P in [('pure2',np.c_[Ub,U**2]),('pure3',np.c_[Ub,U**2,U**3]),('odd3',np.c_[Ub,U**3]),('quad+pure3',np.c_[poly(U,2),U**3])]:
  E=P@fit(P,V)-V;print(label,'rms',np.mean(E**2)**.5,'loo',np.mean(abs(E)/(1-np.diag(P@np.linalg.pinv(P)))[:,None]));
 print('v transformed to estimated linear basis:',(V-fit(Ub,V)[0]) @ np.linalg.inv(fit(Ub,V)[1])) if False else None
 for fun in [np.sin,np.tanh,lambda a:a/(1+a*a),np.sinh,np.cos]:
  for combo in (0,1):
   def solve(p,ret=False):
    P=np.c_[Ub,fun(U*p[:4]+p[4:])] if combo else np.c_[np.ones(len(U)),fun(U*p[:4]+p[4:])]
    return (P@fit(P,V)-V).ravel()
   for mag in [0.65,2.,3.5] if fun==np.sin else [0.65]:
    rr=least_squares(solve,np.r_[[mag]*4,[.1]*4],max_nfev=350); print(fun.__name__,combo,np.mean(rr.fun**2)**.5,rr.x.round(3))
for label,F in [('z',z),('w',w)]:
 print('\n',label)
 for kind,P in [('aff',poly(F,1)),('squares',np.c_[poly(F,1),F**2]),('cubes',np.c_[poly(F,1),F**3]),('sq+cube',np.c_[poly(F,1),F**2,F**3]),('cross+cube',np.c_[poly(F,2),F**3]),('degree3',poly(F,3))]:
  C=fit(P[:52],y[:52]);print(kind,cv(P),np.mean(abs(P[:52]@C-y[:52])),C.round(4))
 F0=poly(F,1)
 for kind,fn in [('sin',np.sin),('tanh',np.tanh),('rational',lambda h:h/(1+h*h)),('cos',np.cos)]:
  def project(p,sub):
   P=np.c_[F0,fn(F@p[:4]+p[4])];return P,fit(P[sub],y[sub]);
  sub=np.arange(52); best=None
  for k in range(10):
   rng=np.random.default_rng(k)
   def err(p):
    P,C=project(p,sub); return P[sub]@C-y[sub]
   rr=least_squares(err,rng.normal(0,1.5,5),max_nfev=350)
   if best is None or np.linalg.norm(rr.fun)<np.linalg.norm(best.fun):best=rr
  P,C=project(best.x,sub);print(kind,cv(P),np.mean(abs(best.fun)),best.x.round(3),C.round(3))
