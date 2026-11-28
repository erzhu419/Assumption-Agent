exec(open('/tmp/stages.py').read().split('train=[',1)[0])
I=[np.arange(52),np.arange(16,72),np.r_[0:27,43:72],np.r_[0:40,56:72]];J=[np.setdiff1d(np.arange(72),i) for i in I]
Q=poly(x,2)
def fitratio(A,d,y,rank):
 def mat(p,A,d):
  return A/(1+(af(d)@p.reshape(5,rank))**2 @ np.ones(rank))[:,None]
 def err(p):
  D=mat(p,A,d);return D@reg(D,y)-y
 rng=np.random.default_rng(8)
 ps=[*list(np.eye(5)[:,1:].T*.5),*rng.normal(0,.3,(4,5))] if rank==1 else rng.normal(0,.3,(5,5*rank))
 opt=min([least_squares(err,p,max_nfev=120,bounds=(-3,3)) for p in ps],key=lambda r:np.linalg.norm(r.fun));return opt.x,reg(mat(opt.x,A,d),y),mat,np.std(opt.fun)
for d,desc in [(x,'x'),(root['i'],'i')]:
 A=poly(d,2)
 for rank in (1,2):
  P=[];R=[]
  for i,j in zip(I,J):
   p,b,m,r=fitratio(A[i],d[i],y[i],rank);P.append(np.mean(abs(m(p,A[j],d[j])@b-y[j])));R.append(r)
  print('rational',desc,rank,np.round(P,5),np.round(R,5),p)

for name,v in [('raw',x),('c1',z['bi1']),('c2',z['bi2']),('dual',np.c_[z['bi1'],z['bi2']])]:
 for base,A in [('quadratic',Q),('joint',np.c_[Q,v,v*v]),('cross',poly(np.c_[x,v],2))] if name!='raw' else [('quadratic',Q),('quartic',poly(x,4))]:
  for link in ('tanh','asinh','sinh','linear'):
   e=[]; methods=[]
   for i,j in zip(I,J):
    scale=np.maximum(.04,A[i].std(0));scale[0]=1;F=A/scale
    U,S,V=np.linalg.svd(F[i],full_matrices=False);h=F[j]@V.T; best=(np.inf,None,None)
    for c in ([1.2,1.45,1.8,2.4,4] if link=='tanh' else [.5,1,2,4] if link!='linear' else [1]):
     if link=='tanh': g=lambda y:np.arctanh(np.clip(y/c,-.999,.999)); inv=lambda z:c*np.tanh(z)
     elif link=='asinh':g=lambda y:np.sinh(y/c);inv=lambda z:c*np.arcsinh(z)
     elif link=='sinh':g=lambda y:np.arcsinh(y/c);inv=lambda z:c*np.sinh(np.clip(z,-8,8))
     else:g=inv=lambda z:z
     yy=g(y[i]);uy=U.T@yy
     for lam in (.001,.01,.07,.3,1,2,4):
      f=S*S/(S*S+lam); loo=yy-(yy-U@(f*uy))/(1-U**2@f); le=np.mean(abs(inv(loo)-y[i]));
      if le<best[0]:best=(le,np.mean(abs(inv(h@(S*uy/(S*S+lam)))-y[j])),(c,lam))
    e.append(best[1]);methods.append(best[2])
   print('link',name,base,link,np.round(e,5),methods)
