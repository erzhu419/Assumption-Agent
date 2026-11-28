import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/chamber_probe.py').read())
for row in log:
 print('log', {k:v for k,v in row.items() if k!='facility_telemetry_trace'},np.round(np.mean(row['facility_telemetry_trace'],0),3))
E=-np.expm1(-t[:,None]/tau); Ep=1/tau
for name,X in [('provided',x),('diag pooled',a._features(tr,s)[1]),('raw',s['coupon_features_raw']),('diag only',a._features(s,s)[1])]:
 X=np.where(np.isfinite(X),X,imp); dr=th[2]+X@th[3:]; print('\nFEATURES',name)
 def fit(A,B):
  Q=np.linalg.lstsq(A,B,rcond=None)[0]; return Q,B-A@Q
 for matrixname,M in [('z + drift',np.c_[hold,z,dr]),('past + drift',np.c_[hold,past,dr]),('past + raw feat',np.c_[hold,past,np.ones(len(dr)),X])]:
  A=np.r_[E,Ep[None,:]]; yy=np.c_[Y,R]; Q,err=fit(M,yy); C,ke=fit(A,Q.T); print(matrixname,'RMSE',np.sqrt(np.mean(err**2)), 'kernel excess',np.sqrt(np.mean((M@ke.T)**2)), 'amps',np.round(C.T,3));
  if matrixname=='z + drift': print('time residual by terms',np.round(ke.T,3))
 # Shared chamber model allowing arbitrary mixing of the two memory branches and a separate drift time/rational basis.
 kernel=lambda q: -np.expm1(-t[:,None]/q)*q
 y=np.c_[Y,R]; A=np.r_[E,Ep[None,:]]; hf=hold[:,None]*(A@th[:2]); y=y-hf
 for kind in ['shared exp','new exp','two exp','rational','exp plus linear','free memory tau']:
  def design(q):
   Am=A
   if kind=='free memory tau':
    Am=np.r_[-np.expm1(-t[:,None]/np.exp(q[:2])),np.exp(-q[:2])[None,:]]; q=q[2:]
   if kind=='shared exp': G=kernel(tau)
   elif kind in ('new exp','free memory tau'): G=kernel(np.exp(q))
   elif kind=='two exp': G=kernel(np.exp(q))
   elif kind=='exp plus linear': G=np.c_[kernel(np.exp(q)),t]
   else: G=(t/(1+t/np.exp(q[0])))[:,None]
   G=np.r_[G,np.ones((1,G.shape[1]))];
   return np.c_[(z[:,:,None,None]*Am[None,None,:,:].transpose(0,1,3,2)).reshape(len(z),4,len(t)+1).transpose(0,2,1).reshape(y.size,4),(dr[:,None,None]*G[None,:,:]).reshape(y.size,-1)]
  def fun(q,pr=False):
   D=design(q); b=np.linalg.lstsq(D,y.ravel(),rcond=None)[0]; e=D@b-y.ravel(); return (b,np.sqrt(np.mean(e**2)),np.quantile(abs(e),[.5,.9,1])) if pr else e
  q={'shared exp':[], 'two exp':[-1.,1.], 'free memory tau':list(np.log(tau))+[0.], 'new exp':[0.], 'rational':[0.], 'exp plus linear':[0.]}[kind]
  if q: q=least_squares(fun,q,bounds=(-6,5)).x
  print(kind,np.round(np.exp(q),3),fun(q,True))
