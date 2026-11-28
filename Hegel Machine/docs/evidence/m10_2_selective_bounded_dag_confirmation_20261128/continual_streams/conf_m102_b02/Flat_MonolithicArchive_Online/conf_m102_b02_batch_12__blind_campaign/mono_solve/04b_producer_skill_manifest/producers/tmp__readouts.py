from discover import *
z=dict(np.load('/tmp/coords.npz')); t=s['train']; Y=np.c_[t['y'],t['traj']]; n=52; out=[]
def ridge(X,Y,Xt,a):
 mu=X.mean(0);sd=np.maximum(X.std(0),1e-8);m=Y.mean(0);Z=(X-mu)/sd; return m+(Xt-mu)/sd@np.linalg.solve(Z.T@Z+a*np.eye(X.shape[1]),Z.T@(Y-m))
def cross(x,y):return (x[:,:,None]*y[:,None,:]).reshape(len(x),-1)
for k,x in z.items():
 for l,w in z.items():
  for mode,d in [('lin',np.c_[x,w]),('self',np.c_[x,w,x*w]),('bilin',np.c_[x,w,cross(x,w)]),('square',np.c_[x,w,cross(x,x),cross(w,w)]),('sat',np.c_[x,w,np.tanh(2.15*x),np.tanh(2.15*w)]),('full',np.c_[x,w,cross(x,w),np.tanh(2.15*x),np.tanh(2.15*w)])]:
   for a in [.003,.1,1]:
    R=ridge(d[:n],Y[:n],d[n:72],a)-Y[n:];v=np.abs(R).mean(0);out.append((v[0],v[1:].mean(),k,l,mode,a))
for i in range(3):print(sorted(out,key=lambda a:[a[0],a[1],max(a[:2])][i])[:16])
