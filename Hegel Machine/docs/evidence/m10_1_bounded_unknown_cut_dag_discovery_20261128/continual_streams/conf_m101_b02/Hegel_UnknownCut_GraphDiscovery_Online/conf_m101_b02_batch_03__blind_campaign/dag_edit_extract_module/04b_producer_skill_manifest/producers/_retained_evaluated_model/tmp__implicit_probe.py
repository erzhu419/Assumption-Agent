exec(open('/tmp/dag_probe.py').read().split("if '--setup'")[0])
p=s['panel_c'];u=p['u']; np.set_printoptions(precision=5,suppress=True,linewidth=150)
for k in ('v1','v2'):
 v=p[k]; print('panel',k,'range',v.min(),v.max());
 for fun in [lambda v:v*v,lambda v:v*abs(v),lambda v:np.sin(v),lambda v:v**3]:
  X=np.c_[L(v),fun(v)]; coef=np.linalg.lstsq(X,u,rcond=None)[0]; print('inv direct',np.max(abs(X@coef-u)),np.linalg.svd(coef[5:],compute_uv=False));
 for W in [v,u]:
  for labs,P in [('squares',W**2),('all',np.column_stack([W[:,i]*W[:,j] for i in range(4) for j in range(i,4)]))]:
   X=np.c_[L(v),P]; coef=np.linalg.lstsq(X,u,rcond=None)[0]; print('inv quad', W is v,labs,np.max(abs(X@coef-u))); 
   if W is v: print(coef)
# Test compact, rotation-invariant polynomial or rational readouts of both calibrated and original observations.
H=-np.expm1(-t[:,None]*np.array([.42,1.8])); Y=np.c_[y,T@np.linalg.pinv(H.T)]; a=np.arange(52);i=np.arange(52,72)
for nm,q in [('z',z),('x',x),('b',b),('f',f),('c',c),('d',d)]:
 for mode in ['linear','quadratic','cubic','pairs','symmetry']:
  if mode=='linear': X=L(q)
  elif mode in ('quadratic','cubic'):
   X=np.column_stack([np.prod(q[:,ids],axis=1) for j in range(3+(mode=='cubic')) for ids in it.combinations_with_replacement(range(4),j)])
  else:
   cols=[]
   for j,k in it.combinations(range(4),2):
    v,w=q[:,j],q[:,k]; cols.extend([np.sin(v+w),np.sin(v-w),np.tanh(v+w),np.tanh(v-w),v*w])
    if mode=='symmetry':cols.extend([v/(1+w*w),w/(1+v*v),np.cos(v+w),np.cos(v-w)])
   X=np.c_[L(q),np.array(cols).T]
  X=X/np.maximum(np.sqrt(np.mean(X[a]**2,axis=0)),1e-9)
  for cap in [1.,1.2,1.5,2.,3.,np.inf]:
   if cap<=abs(y).max(): continue
   Z=Y.copy();Z[:,0]=y if cap==np.inf else np.arctanh(y/cap)
   errors=[]
   for lam in [.001,.03,.15,.5]:
    P=X@np.linalg.solve(X[a].T@X[a]+lam*np.eye(X.shape[1]),X[a].T@Z[a]); P[:,0]=P[:,0] if cap==np.inf else cap*np.tanh(P[:,0]);e=P-Y
    errors.append((abs(e[i,0]).mean(),abs(e[i,1:]@H.T).mean(),lam))
   if min(e[0] for e in errors)<.07 or min(e[1] for e in errors)<.065:print('strong',nm,mode,cap,min(errors),min(errors,key=lambda v:v[1]))
