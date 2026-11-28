exec(open('/tmp/explore.py').read().split('a=s[')[0])
p=s['panel_a']; keep=np.linalg.norm(p['v']-p['u'],axis=1)<2; scale=np.array([ls(poly(p['u'][keep,j:j+1],1),p['v'][keep,j]) for j in range(4)]); w=(x-scale[:,0])/scale[:,1]
p=s['panel_b']; B=ls(poly(p['u'],1),p['v']); b=poly(w,1)@B
p=s['panel_c']; u=p['u']; C=np.c_[p['v1'],p['v2']]; K=poly(u,1)
for F in [poly(u,2),np.c_[K,np.sin(u),np.cos(u)],np.c_[K,np.tanh(u),u**2],np.c_[K,np.sin(2*u),np.cos(2*u)]]:
 L=ls(F[:18],C[:18]);print('C',F.shape,np.std(F@L-C,axis=0),np.std((F@L-C)[18:],axis=0))
print('C quadratic coeff',ls(poly(u,2),C).T)
for name,X in [('x',x),('inverse A',w),('inverse A then B',b)]:
 P=poly(X,2); mode=np.linalg.svd(z,full_matrices=False)[2][0]; Z=np.c_[y,z@mode]; L=ls(P,Z); R=Z-P@L;print(name,'quad',L.T,'rms',np.std(R,axis=0));
 for j in range(2):
  H=np.zeros((4,4));
  for (i,k),v in zip(combinations_with_replacement(range(4),2),L[5:,j]):H[i,k]=H[k,i]=v/(2 if i!=k else 1)
  print('quad components',np.linalg.eigh(H));
 from scipy.spatial.distance import cdist
 d=cdist(X,X,'sqeuclidean'); eye=np.eye(n)
 for k in [.8,1.5,2.5,4]:
  for form in ['se','se-dot','matern']:
   G=np.exp(-d/(2*k*k));r=np.sqrt(5*d)/k
   if form=='se-dot':G*=1+X@X.T
   if form=='matern':G=(1+r+r*r/3)*np.exp(-r)
   M=P@P.T+5*G; tr=slice(56);te=slice(56,None)
   print(name,k,form,np.mean(abs(M[te,tr]@np.linalg.solve(M[tr,tr]+.001*eye[tr,tr],Z[tr])-Z[te]),0))
