from graph import *
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=150)
_,_,V=np.linalg.svd(T[:52],full_matrices=False); Y2=np.c_[y,T @ (V[0]*np.sign(V[0,-1])),T @ V[1]]
for n in ['r','a','f','rb','ab','fb','rv1','rv2','fbv1','fbv2']:
 w=Z[n]; W=soft(w,.22); F=basis(W,3); C=ls(F[:52],Y2[:52]); print(n,'c1',C[:5,:2].T,'c2',C[5:15,:2].T,'fit',np.std(Y2[:52]-F[:52]@C,0),'val',np.abs(Y2[52:]-F[52:]@C).mean(0))
for n in ['r','a','f','rb','ab','fb','rv1','rv2','fbv1','fbv2']:
 w=Z[n]; R=[]
 for g in [0,.09,.22,.5,1,2]:
  for transform in [soft,fun['tanh'],fun['atan']]:
   z=transform(w,g) if g else w
   A1=af(z); A2=basis(z,2); A3=basis(z,3); H=np.c_[z*z,z**3]; M=np.c_[A2,z**3]; cross=A2[:,[6,7,8,10,11,13]]
   opts=[('linear',A1),('quadratic',A2),('add3',np.c_[A1,H]),('mix3',M),('cross',np.c_[A1,cross]),('full3',A3)]
   for k,F in opts:
    for v in [0,.22,.5,1.25]:
     for den in [np.sum(z*z,1),np.sum(w*w,1),np.sum(z,1)**2,np.prod(z,1)**2, (z[:,0]-z[:,1])**2,(z[:,2]-z[:,3])**2]:
      if v==0 and den is not None and not np.array_equal(den,np.sum(z*z,1)): continue
      G=F/(1+v*den[:,None]); G=np.c_[np.ones(len(F)),G[:,1:]] if not v else np.c_[np.ones(len(F)),G]
      a=score(G,Y,.003);R.append((a,(n,g,transform.__name__,k,v,tuple(np.round(den[:2],3)))))
 print(n,'y',min(R,key=lambda r:r[0][2]),'t',min(R,key=lambda r:r[0][3]),'y52',min(R,key=lambda r:r[0][0]),flush=True)
