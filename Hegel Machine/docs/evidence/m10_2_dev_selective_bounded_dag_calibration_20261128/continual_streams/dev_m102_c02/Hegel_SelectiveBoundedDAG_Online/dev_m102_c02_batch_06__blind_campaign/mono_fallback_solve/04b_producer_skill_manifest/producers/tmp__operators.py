exec(open('/tmp/probe.py').read().split("a=s['panel_a']")[0]); from scipy.optimize import least_squares
p=s['panel_a']; good=np.linalg.norm(p['v'],axis=1)<5; u,v=p['u'][good],p['v'][good]
for k in range(4):
 print('A coord',k,'ols',fit(poly(u,1),v[:,k]),'v minmax',v[:,k].min(),v[:,k].max())
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u'];
 for lab,y in p.items():
  if lab=='u':continue
  mask=np.linalg.norm(y,axis=1)<5; a=u[mask]; b=y[mask]; X=poly(a,1); M=fit(X,b)
  print(k,lab,'RMSE by operator')
  forms={'cube':lambda w:w**3,'tanh':np.tanh,'sin':np.sin,'sat':lambda w:w/np.sqrt(1+w*w),'rat':lambda w:w/(1+w*w),'square':lambda w:w*w,'soft':lambda w:np.logaddexp(w,0)}
  for n,f in forms.items():
   vals=[]
   for i in range(4):
    z=b[:,i]; m=M[:,i]; errs=[]
    def los(c): return c[0]+c[1]*f(X@c[2:])-z
    def add(c): return X@c[:5]+c[5]*f(X@c[6:])-z
    for fun,q in [(los,np.r_[0.,1.,m]),(add,np.r_[m,1.,m])]:
     res=least_squares(fun,q,max_nfev=280); errs.append(np.std(res.fun))
    vals.append(errs)
   print(n,np.array(vals).T)
  for typ,Z in [('diag3',np.c_[X,a*a,a**3]),('cubenooff',np.c_[a,a**3]),('sqnooff',np.c_[a,a*a])]:
   m=fit(Z,b); print(typ,np.std(Z@m-b,0),m if k=='panel_a' or (k=='panel_c' and lab=='v2') else '')
