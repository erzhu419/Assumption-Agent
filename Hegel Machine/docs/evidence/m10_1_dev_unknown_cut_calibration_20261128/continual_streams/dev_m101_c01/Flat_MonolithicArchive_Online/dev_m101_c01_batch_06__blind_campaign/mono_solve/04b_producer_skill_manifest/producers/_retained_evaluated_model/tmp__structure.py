exec(open('/tmp/explore.py').read().split('a=s[')[0])
from scipy.linalg import svd
from scipy.optimize import minimize
p=s['panel_a']; keep=np.linalg.norm(p['v']-p['u'],axis=1)<2; A=ls(poly(p['u'][keep],1),p['v'][keep]); ax=poly(x,1)@A
p=s['panel_b']; B=ls(poly(p['u'],1),p['v']); bx=poly(ax,1)@B; bx0=poly(x,1)@B
Q,S,V=svd(z,full_matrices=False); print('temporal V',V[:3],S[:4]); U=Q[:,:2]*S[:2]; W=np.c_[y,U]; F=poly(ax,2); E=W-F@ls(F,W); L=ls(F,W); print('affine+quad coef',L.T); inds=list(combinations_with_replacement(range(4),2))
for j in range(3):
 H=np.zeros((4,4));
 for (a,b),c in zip(inds,L[5:,j]): H[a,b]=H[b,a]=c/(2 if a!=b else 1)
 print('eig',np.linalg.eigh(H),'H in B',np.linalg.inv(B[1:])@H@np.linalg.inv(B[1:]).T)
for name,X in [('raw',x),('A',ax),('B',bx),('Braw',bx0)]:
 for label,fun in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('soft',lambda x:x/(1+x*x)),('cube',lambda x:x**3)]:
  cols=[]
  for a in range(4):
   for b in range(a,4):
    for sign in ([0] if a==b else [-1,1]):
     d=X[:,a]+sign*X[:,b]; cols.extend([(f'{a},{b},{sign},{f}',fun(f*d)) for f in ([1] if label=='cube' else [.5,1,1.5,2,3])])
  Z=np.stack([c for _,c in cols],axis=1); R=Z-F@ls(F,Z); corr=abs(R.T@E)/np.maximum(np.linalg.norm(R,axis=0)[:,None],1e-10)/np.linalg.norm(E,axis=0); print(name,label,[(cols[k][0],round(corr[k,j],4)) for j,k in enumerate(corr.argmax(0))]);
print('summary reconstructed',y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std())
