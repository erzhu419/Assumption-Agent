exec(open('/tmp/spatial.py').read().split('fs={}')[0]); np.set_printoptions(precision=5,suppress=True,linewidth=145)
u=b['u']; v=b['v'];
funcs={
'cube':(lambda u,a:u**3,[0]), 'signedpow':(lambda u,a:u*abs(u)**a[0],[2.]), 'rational3':(lambda u,a:u**3/(1+a[0]*u*u),[.22]), 'rational3p':(lambda u,a:u**3/(1+a[0]*abs(u)**a[1]),[.22,2]), 'rational3k':(lambda u,a:u**3/(1+a[0]*u*u)**a[1],[.22,1]), 'rational34':(lambda u,a:u**3/(1+a[0]*u**4),[.22]), 'exp3':(lambda u,a:u**3*np.exp(-a[0]*u*u),[.22]), 'sin':(lambda u,a:np.sin(a[0]*u),[2.5]), 'tanh':(lambda u,a:np.tanh(a[0]*u),[1.25]), 'softpow':(lambda u,a:u*(1+(a[0]*abs(u))**a[1])**a[2],[1.25,5.,.15]), 'root':(lambda u,a:u*np.sqrt(1+a[0]*u*u),[1.]), 'atan':(lambda u,a:np.arctan(a[0]*u),[1.25]), 'asinh':(lambda u,a:np.arcsinh(a[0]*u),[1.25]), 'clip':(lambda u,a:np.clip(u,-a[0],a[0]),[.7]), 'shift_rat3':(lambda u,a:(u+a[1])**3/(1+a[0]*(u+a[1])**2),[.22,0]), 'soft3':(lambda u,a:u**3/(1+a[0]*abs(u)),[.22]), 'soft3k':(lambda u,a:u**3/(1+a[0]*abs(u))**a[1],[.22,1])}
results=[]
for name,(fun,init) in funcs.items():
 for linear in (False,True):
  def feat(u,a):return np.c_[one(u),u,fun(u,a)] if linear else np.c_[one(u),fun(u,a)]
  with np.errstate(all='ignore'):
   try:
    err,params,op=fitfun(u,v,feat,init)
   except (ValueError,np.linalg.LinAlgError):continue
  B=op(x); err1=cv(np.c_[one(x),B]); err2=cv(poly(B,2)); print(name,linear,round(err,7),params,'lin',err1,'quad',err2); results.append((err,name,linear,params,op,fun))
for err,name,linear,params,op,fun in sorted(results,key=lambda a:a[0])[:5]:
 B=op(x); F=np.c_[one(x),B]; E=y-F@ls(F,y); print('\nbest',err,name,params,'coefs',ls(F,np.c_[y,t[:,0],t[:,-1]]).T); print('resid rms',np.std(E),'resid x powers rows',(np.corrcoef(np.c_[E,x,x*x,x**3,B*B].T)[0,1:]).reshape(-1,4)); print('raw vs inv',cv(np.c_[one(x),op(tr['x'])]), cv(np.c_[one(x),op(tr['x']*R[0]+R[1])]))
 print('panel coeff',ls(np.c_[one(u),u,fun(u,params)],v).T)
 print('feature trials: x,B; x^2,B; x^3,B; x*B; B^3',*[cv(f) for f in [np.c_[F,x],np.c_[F,x*x],np.c_[F,x**3],np.c_[F,x*B],np.c_[F,B**3]]]); print('quad y coefs', ls(poly(B,2),y))
