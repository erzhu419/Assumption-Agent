from discover import *
from scipy.optimize import least_squares
from scipy.linalg import qr
from scipy.spatial.distance import cdist
tr=s['train']; X=cal(tr['x'],True); U=s['panel_b']['u']; V=s['panel_b']['v']; f=lambda x:np.c_[one(x),np.sin(2.8*x)]; Z=f(X)@fit(f(U),V); W=X-.4*np.sin(2.8*X); H=one(Z); Y=np.c_[tr['y'],tr['traj']]; print('panel B support',U.min(0),U.max(0));
sp=[np.arange(72)]+[np.random.default_rng(k).permutation(72) for k in range(5)]
report=lambda l:np.round(np.array(l)[[0]].tolist()+[np.mean(l,axis=0),np.max(l,axis=0)],5)
def err(p,v):
 e=abs(p-Y[v]);return [e[:,0].mean(),e[:,1:].mean()]
D={'Z':Z,'X':X,'raw':tr['x'],'W':W,'ZX':np.c_[Z,X]}
for nm,A in D.items():
 A=A/A.std(0); scores={k:[] for k in ['matern','gauss','additive','pair','lin','tanh','sin','rat','sq','cube']}
 d=cdist(A,A,'sqeuclidean')/A.shape[1]; d1=(A[:,None,:]-A[None,:,:])**2; grids=[.22,.4,.7,1.2,2.]; kernels={}
 for kind in ['matern','gauss','additive','pair']:
  kernels[kind]=[]
  for w in grids:
   E=np.exp(-d1/(2*w*w)); m=np.sqrt(3*d)/w
   K={'matern':(1+m)*np.exp(-m),'gauss':np.exp(-d/(2*w*w)),'additive':E.mean(2),'pair':(E.sum(2)**2-(E**2).sum(2))/(A.shape[1]*(A.shape[1]-1))}[kind];kernels[kind].append(K)
 for ix in sp:
  a,b=ix[:56],ix[56:]; Q=np.linalg.qr(H[a])[0]; Ry=Y[a]-Q@(Q.T@Y[a]); _,_,vh=np.linalg.svd(Ry,full_matrices=False); y=Ry@vh[0]; scores['lin'].append(err(H[b]@fit(H[a],Y[a]),b))
  for kind,Ks in kernels.items():
   vals=[]
   for K in Ks:
    for r in [.003,.03,.3,3.]:
     M=np.block([[K[np.ix_(a,a)]+r*np.eye(len(a)),H[a]],[H[a].T,np.zeros((5,5))]]); Mi=np.linalg.inv(M); c=Mi@np.r_[Y[a],np.zeros((5,9))]; e=c[:len(a)]/Mi.diagonal()[:len(a),None]; vals.append(((abs(e[:,0]).mean()+abs(e[:,1:]).mean()*1.7),err(np.c_[K[np.ix_(b,a)],H[b]]@c,b)))
   scores[kind].append(min(vals,key=lambda v:v[0])[1])
  if nm=='ZX':continue
  for kind,g in [('sin',np.sin),('tanh',np.tanh),('rat',lambda z:z/(1+z*z)),('sq',lambda z:z*z),('cube',lambda z:z*z*z)]:
   def obj(w):
    v=g(A[a]@w[:-1]+w[-1]);v-=Q@(Q.T@v);return y-v*(y@v)/(v@v+1e-12)
   rg=np.random.default_rng(17); found=[]
   for w in np.r_[np.eye(5),-np.eye(5),rg.normal(size=(23,5))*1.2]:
    res=least_squares(obj,w,bounds=(-7,7),max_nfev=75); found.append((np.mean(res.fun**2),res.x))
   r,w=min(found,key=lambda p:p[0]); F=np.c_[H,g(A@w[:-1]+w[-1])];scores[kind].append(err(F[b]@fit(F[a],Y[a]),b));
   if ix is sp[0]:print('ridge',nm,kind,'fit',round(r,5),'axis',w.round(3))
 for k,l in scores.items():
  if l:print(nm,k,report(l).tolist())
