from operators import *
from itertools import combinations
from scipy.linalg import qr
end = z @ (np.linalg.svd(z,full_matrices=False)[2][0]/np.linalg.svd(z,full_matrices=False)[2][0,-1]); g=np.linalg.svd(z,full_matrices=False)[2][0]; end=z@g*g[-1]; Y2=np.c_[y,end]
B=aff(x)@bb; Br=aff(raw)@bb
P2=PolynomialFeatures(2).fit_transform(x)
def cross(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def check(F,tr,te,j,ridge=0.):
 U,sv,V=np.linalg.svd(F[tr],full_matrices=False); v=sv/(sv*sv+ridge); v[sv<1e-8]=0
 co=V.T@(v*(U.T@Y2[tr,j])); return np.mean(abs(F[te]@co-Y2[te,j]))
outputs=[[],[]]
for kn,X in [('raw',raw),('A',x),('B(A)',B),('B(raw)',Br),('B inverse',(x-bb[0])@np.linalg.inv(bb[1:]))]:
 for p in (2,3,4):
  poly=PolynomialFeatures(p); M=poly.fit_transform(X); powers=poly.powers_; degree=powers.sum(1); nterms=(powers>0).sum(1)
  sets={'full':degree<=p,'pair':nterms<=2,'axis_high':(degree<=2)|(nterms<=1),'biquad':(degree<=2)|((nterms<=2)&(powers.max(1)<=2)),'even_high':(degree<=3)|((powers%2==0).all(1))}
  for key,mask in sets.items():
   if p<4 and key not in ('full','pair','axis_high'):continue
   for j in range(2):
    F=M[:,mask]; errors=[check(F,tr,te,j,1e-6) for tr,te in folds]; outputs[j].append((max(errors),np.mean(errors),errors[0],kn,p,key,F.shape[1],np.std(Y2[:,j]-F@fit(F,Y2[:,j]))))
 for nm,h in [('C1',c1map(X)),('C2',c2map(X)),('square',X**2),('cross',PolynomialFeatures(2).fit_transform(X)[:,5:])]:
  Z2=PolynomialFeatures(2).fit_transform(h); Q=np.c_[np.ones(len(x)),X,h]
  designs={'quad':Z2,'p2+quad':np.c_[P2,Z2],'p2+quad+cross':np.c_[P2,Z2,cross(X,h)],'p2+diag':np.c_[P2,h,h*h,X*h if X.shape==h.shape else cross(X,h)],'p2+quart':np.c_[P2,cross(h,h)]}
  for k,F in designs.items():
   for j in range(2):
    for l in [1e-5,1e-3,.01]:
     errors=[check(F,tr,te,j,l) for tr,te in folds]; outputs[j].append((max(errors),np.mean(errors),errors[0],kn,nm,k,F.shape[1],np.std(Y2[:,j]-F@fit(F,Y2[:,j])),l))
for j,rows in enumerate(outputs):
 print('TARGET',j);print('TOP CV');[print(r) for r in sorted(rows,key=lambda a:a[:2])[:17]];print('TOP FIT SMALL');[print(r) for r in sorted([r for r in rows if r[6]<50],key=lambda a:a[7])[:12]]
