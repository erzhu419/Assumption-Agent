from probe import *
V1=lambda a:np.c_[np.ones(len(a)),a[:,0]*a[:,1],a[:,0]*a[:,3],a[:,1]*a[:,2],a[:,2]*a[:,3]]
V2=lambda a:np.c_[np.ones(len(a)),np.cos(2.4*a)]
g=V1(z)@ls(V1(u),v1); h=V2(z)@ls(V2(u),v2); F=np.c_[g,h]; N=len(F)
print('targets',[(k,np.shape(v),np.min(v),np.max(v)) for k,v in T.items()]); print('design spread',F.std(0));
sq=lambda a: np.column_stack([a[:,i]*a[:,j] for i,j in it.combinations_with_replacement(range(a.shape[1]),2)])
cross=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
groups={'gquad':sq(g),'hquad':sq(h),'cross':cross(g,h),'z':z,'zquad':sq(z),'abs':abs(F),'tanh':np.tanh(F),'sin':np.sin(F),'cos':np.cos(F),'softplus':np.logaddexp(0,F),'cube':F**3,'matched':g*h,'square':F**2,'zreadout':cross(z,F),'norm_z':np.sum(z*z,axis=1,keepdims=True)}
res=[]
for n,a in groups.items():
 for m,b in list(groups.items())+[('-',np.empty((N,0)))]:
  if n==m or a.shape[1]+b.shape[1]>44: continue
  D=np.c_[F,a,b]; v=ev(D); res.append((v.max(),n,m,v[0],v.max(0)))
for r in sorted(res,key=lambda t:t[0])[:27]: print(r)
# Sparse joint residual screening, with selection repeated inside each split.
G=np.column_stack(list(groups.values())); labels=[f'{k}:{i}' for k,a in groups.items() for i in range(a.shape[1])]
Y=np.c_[T['y'],T['traj']]; Y=Y.reshape(N,-1)
from scipy.linalg import qr
for a,b in [*folds,(np.arange(N),np.arange(N))]:
 X=np.c_[np.ones(N),F]; sel=[]; yt=Y[a]; print('fold',len(a),end=' ')
 for j in range(18):
  Q=np.linalg.qr(X[a])[0]; L=G[a]-Q@(Q.T@G[a]); R=yt-Q@(Q.T@yt); sc=np.linalg.norm(L.T@(R/yt.std(0)),axis=1)/np.maximum(np.linalg.norm(L,axis=0),1e-10); sc[sel]=-1; i=np.argmax(sc); sel.append(i); X=np.c_[X,G[:,i]]
  if j in [0,1,3,5,7,11,17]:
   r=np.mean(abs(X[b]@ls(X[a],yt)-Y[b]),axis=0); print(j+1,labels[i],np.round([r[0],r[1:].mean()],4),end='; ')
 print('\n',[labels[i] for i in sel])
