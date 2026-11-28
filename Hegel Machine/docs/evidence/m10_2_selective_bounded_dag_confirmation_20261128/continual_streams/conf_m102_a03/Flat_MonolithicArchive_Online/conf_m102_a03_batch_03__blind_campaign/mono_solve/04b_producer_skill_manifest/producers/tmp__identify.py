exec(open('/tmp/probe.py').read().split('U,D,V=')[0])
for k in [1,2,3]:
 d=poly(u,k); print('poly',k,np.linalg.norm(d@fit(d,v)-v))
funcs={'sin':np.sin,'atan':np.arctan,'tanh':np.tanh,'cos':np.cos,'erf':__import__('scipy').special.erf}
for odd in ('sin','atan','tanh','erf'):
 for even in ('square','cos','none'):
  for linear in (False,True):
   def des(x,p):
    cols=[np.ones((len(x),1))];
    if linear: cols.append(x)
    cols.append(funcs[odd](p[0]*x))
    if even=='square': cols.append(x*x)
    if even=='cos': cols.append(np.cos(p[1]*x))
    return np.column_stack(cols)
   def rr(p):
    d=des(u,p);return (d@fit(d,v)-v).ravel()
   sol=min((least_squares(rr,p,bounds=(.03,10.),max_nfev=200) for p in ([.7,1.3],[1.8,2.],[3.8,4.],[5.8,2.])),key=lambda q:q.cost)
   p=sol.x; d=des(u,p); coef=fit(d,v); w=des(z,p)@coef
   quad=poly(w,2); err=abs(quad[52:]@fit(quad[:52],Y[:52,0])-Y[52:,0]).mean()
   print(odd,even,linear,p,np.std(rr(p)),err)
   if np.std(rr(p))<.01: print('coef',coef)
for name,w in [('B2',project(lambda q,p:np.c_[aff(q),np.arctan(p[0]*q),np.sin(p[1]*q)],[1.2,1.9])),('B5',project(lambda q,p:np.c_[aff(q),np.sin(p[0]*q),q*q],[1.2,1.9]))]:
 d=aff(w); C=fit(d,Y[:,1:]); print(name,'trajectory spatial',C[:,[0,-1]]); print('scalar quad',fit(poly(w,2),Y[:,0])); q=np.linalg.qr(d[:52])[0]; targ=q.T@Y[:52,1:];
 def basis(p):return -np.expm1(-np.outer(t,p))
 def r(p):
  f=basis(p);return (f@fit(f,targ.T)-targ.T).ravel()
 p=least_squares(r,[.4,1.8],bounds=(.001,20.)).x; f=basis(p); print('rates',p,'projnoise',np.std(r(p)),'traj CV',abs(d[52:]@fit(d[:52],Y[:52,1:])@f@np.linalg.pinv(f)-Y[52:,1:]).mean())
