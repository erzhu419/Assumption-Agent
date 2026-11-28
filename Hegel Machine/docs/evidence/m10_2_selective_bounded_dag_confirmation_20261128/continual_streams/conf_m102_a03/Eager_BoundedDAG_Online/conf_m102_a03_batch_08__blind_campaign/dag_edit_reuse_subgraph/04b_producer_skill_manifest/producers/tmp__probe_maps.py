exec(open('/tmp/probe_panels.py').read().split('for k,panel')[0])
b=s['panel_b']; c=s['panel_c']; print('panel sizes',len(b['u']),len(c['u'])); print('time modes',np.linalg.svd(T,full_matrices=False)[2][:3]);
for name,x,v in [('B',b['u'],b['v']),('C1',c['u'],c['v1']),('C2',c['u'],c['v2']),('C12',c['v1'],c['v2'])]:
 print(name)
 for k in range(1,4):
  X=poly(x,k); A=reg(X,v); err=np.mean(abs(X@A-v),axis=0); loo=[]
  for i in range(len(x)):
   j=np.arange(len(x))!=i; loo.append(abs(X[i]@reg(X[j],v[j])-v[i]));
  print(k,'err',err,'loo',np.mean(loo,0))
  if k==2:
   print('terms',list(itertools.combinations_with_replacement(range(4),2))); print(A)
 for deg in (1,2,3):
  for lam in ((0,.01,.1) if deg==3 else (0,)):
   X=poly(x,deg); A=np.linalg.lstsq(np.r_[X,np.sqrt(lam)*np.eye(X.shape[1])],np.r_[v,np.zeros((X.shape[1],4))],rcond=None)[0]; w=poly(u,deg)@A; a,b1,d,e=w.T; Xs=np.c_[np.ones(n),w,a*d+b1*e]; print('readouts',deg,lam,scores(Xs,y),scores(poly(w,2),y),scores(poly(w,1),T),scores(poly(w,2),T))
print('coordinate alternatives: diagonal cubes, interaction, separable cubic');
for qname,q in [('u',u),('z',z)]:
 for X in [np.c_[poly(q,1),q**3],np.c_[poly(q,2),q**3],np.c_[poly(q,1),q**2,q**3],np.c_[poly(q,2),q**3,q**5],np.c_[poly(q,2),np.tanh(q)]]:
  print(qname,X.shape,scores(X,y),scores(X,T))
