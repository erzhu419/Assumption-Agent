from discover import *
from scipy.optimize import least_squares
from scipy.linalg import qr
P=get(); tr=s['train']; X=cal(tr['x'],True); Z=P['panel_bv'](X); T=tr['traj']; y=tr['y']; Y=np.c_[y,T]; u,d,v=np.linalg.svd(T,full_matrices=False); S=T@v[0]; G=np.c_[y,S]; t=s['eval_times']; F=one(Z); R=G-F@fit(F,G); print('res corr',np.corrcoef(R.T)); print('res PCA',np.linalg.svd(R,full_matrices=False)[1:]); print('overall time basis',v[:4].round(4)); print('projected time-residual singular',np.linalg.svd(T-F@fit(F,T),compute_uv=False)); print('time residual relation y',np.corrcoef((Y-F@fit(F,Y)).T)[0]); print('X range',X.min(0),X.max(0)); print('Xtest range',cal(s['test']['x'],True).min(0),cal(s['test']['x'],True).max(0))
D={'B':F,'BX':np.c_[F,X],'BXsin':np.c_[F,X,np.sin(2.8*X)],'Braw':np.c_[F,tr['x']],'BatanX':np.c_[F,np.arctan(2.1*X)],'BC1':np.c_[F,P['panel_cv1'](Z)],'BC2':np.c_[F,P['panel_cv2'](Z)],'BC1X':np.c_[F,P['panel_cv1'](X)],'BC2X':np.c_[F,P['panel_cv2'](X)],'C1B':one(P['panel_cv1'](Z)),'C2B':one(P['panel_cv2'](Z)),'BXcub':np.c_[F,X,X**3]}
B=X-.4*np.sin(2.8*X)
for nm,V in [('Z',Z),('B0',B),('X',X),('raw',tr['x'])]:
 for fn in [('atan',lambda a:np.arctan(2.1*a)),('sin',lambda a:np.sin(2.8*a)),('sinh',np.sinh),('sq',lambda a:a*a),('cube',lambda a:a**3),('abs',np.abs)]:D[nm+fn[0]]=np.c_[F,fn[1](V)]
 for k in [2,3]: D[nm+str(k)]=np.c_[F,poly(V,k)[:,5:]]
 for c in (1.,.5):
  W=np.column_stack([V[:,i]+sg*c*V[:,j] for i in range(4) for j in range(i+1,4) for sg in [-1,1]])
  for fname,w in [('cos',np.cos(2.8*W)),('sin',np.sin(2.8*W)),('atan',np.arctan(2.1*W)),('cube',W**3),('abs',np.abs(W))]: D[nm+fname+str(c)]=np.c_[F,w]
sp=[np.arange(72)]+[np.random.default_rng(k).permutation(72) for k in range(5)]; out=[]
for nm,H in D.items():
 e=[]
 for ids in sp:
  a,b=ids[:56],ids[56:]; err=abs(H[b]@fit(H[a],Y[a])-Y[b]);e.append([err[:,0].mean(),err[:,1:].mean()]);
 e=np.array(e); out.append((nm,e[0],e.mean(0),e.max(0)))
for j in range(2):
 print('sorted',j)
 for nm,a,b,c in sorted(out,key=lambda row:row[2][j])[:18]:print(nm,a.round(5),b.round(5),c.round(5))
for name,H in [('Z',Z),('B0',B),('X',X)]:
 for i in range(4):
  W=np.c_[H[:,i],R]; print(name,i,W[np.argsort(W[:,0])][::6].T.round(2).tolist())
print('y,state simultaneous linear harmonic',fit(np.c_[one(X),np.sin(2.8*X)],G).round(4))
