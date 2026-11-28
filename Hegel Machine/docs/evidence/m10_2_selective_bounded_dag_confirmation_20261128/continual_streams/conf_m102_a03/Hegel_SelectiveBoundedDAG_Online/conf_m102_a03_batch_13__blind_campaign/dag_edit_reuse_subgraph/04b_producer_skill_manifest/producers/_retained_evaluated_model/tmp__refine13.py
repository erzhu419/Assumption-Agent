from read13 import *
r=b[:,0]*b[:,2]+b[:,1]*b[:,3]; diff=b[:,0]*b[:,2]-b[:,1]*b[:,3]
for name,A in [('bz',np.c_[aff(b),z]),('bz+r',np.c_[aff(b),z,r]),('bz+quad',np.c_[poly(b,2),z]),('bz+r+diff',np.c_[aff(b),z,r,diff]),('bz+r+bb',np.c_[aff(b),z,r,r[:,None]*b]),('bz+r+zz',np.c_[aff(b),z,r,r[:,None]*z])]:
 print(name,scores(A,y,.0001),fit(A,y).round(4),np.mean(abs(y-A@fit(A,y))))
 A2=np.c_[A,r**3]; print('pair cube',scores(A2,y,.0001))
A=np.c_[aff(b),z]; P=np.c_[A,r]; R=y-P@fit(P,y); print('residual',np.c_[r,R][np.argsort(r)].round(3).tolist())
for expr,q in [('r',r),('diff',diff),('b0-b1',b[:,0]-b[:,1]),('b0+b1',b[:,0]+b[:,1]),('b2-b3',b[:,2]-b[:,3]),('b2+b3',b[:,2]+b[:,3])]:
 for mult in (.1,.3,1,3,10):
  F=np.c_[A,r/(1+mult*q*q)]; print(expr,mult,scores(F,y,.0001))
for c1,c2 in ((.25,1.),(1.,.25),(1.,1.),(0.,1.),(1.,0.)):
 for coef in (.001,.01,.1,1.):
  G=np.c_[aff(b),poly(b,2,2), c1*poly(b,3,3),c2*z]; es=[]
  for a,j in splits:
   H=G[a]; c=np.linalg.solve(H.T@H+coef*np.diag([0]+[1]*(H.shape[1]-1)),H.T@y[a]);es.append(np.mean(abs(G[j]@c-y[j])))
  print('penalty',c1,c2,coef,np.round([es[0],np.median(es),max(es)],4))
