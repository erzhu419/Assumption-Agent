exec(open('/tmp/probe.py').read().split('for k in')[0])
u=c['u']; v1=c['v1']; v2=c['v2']; au=basis(u,1); bu=au@fb
for name,F in [('diagonal',np.c_[au,u*u]),('diag+cubic',np.c_[au,u*u,u**3]),('diag+quartic',np.c_[au,u*u,u**3,u**4]),('diagB',np.c_[au,bu*bu]),('both',np.c_[au,u*u,bu*bu]),('v1v2',np.c_[au,v2])]:
 H=F@np.linalg.pinv(F); E=np.c_[v1,v2]-H@np.c_[v1,v2];print(name,'loo',np.std(E/(1-np.diag(H))[:,None],axis=0).round(5),'fit',np.std(E,axis=0).round(5))
print('v1 diagonal',fit(np.c_[au,u*u],v1).round(3));print('v2 diagonal',fit(np.c_[au,u*u],v2).round(3)); print('y',np.quantile(y,[0,.05,.25,.5,.75,.95,1])); print('base residual', np.std(Y-basis(x,2)@fit(basis(x,2),Y),axis=0));
for name,U in [('raw',x),('cal',xa),('b',basis(x,1)@fb),('ab',basis(xa,1)@fb),('v1quad',np.c_[basis(x,1),x*x]@fit(np.c_[au,u*u],v1))]:
 items=[]
 for k in [0,.06,.12,.24]:
  W=sq(U,k); D=cdist(W,W,'sqeuclidean'); g=W@W.T/4
  for length in [2,3,4,5,6,7,8,9,10,12]:
   for a in [0,.1,1]:
    K=np.exp(-D/length)+a*(1+g)**2
    for l in [2e-5,1e-4,3e-4]:
     q=show(K,l);items.append((q[2]+.5*q[4],name,k,length,a,l,q))
 print('robust', sorted(items,key=lambda r:r[0])[:4]);print('official', sorted([r for r in items if r[-1][0]<.038],key=lambda r:r[0])[:2])
